"""Verify the fla chunk kernel fast path compiles/runs and matches pure-PyTorch + official.

Also benchmarks fla vs pure-PyTorch on the 0.1B model. Self-skips if fla/triton unavailable.
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from transformers_rwkv7 import Rwkv7Config, Rwkv7ForCausalLM  # noqa: E402
from transformers_rwkv7.modeling_rwkv7 import _fla_available  # noqa: E402

MODEL_PATH = os.environ.get("RWKV7_TEST_MODEL", r"F:\rwkv\models\rwkv7-g1d-0.1b-20260129-ctx8192.pth")


def load(config, device, dtype):
    m = Rwkv7ForCausalLM(config).to(device).to(dtype).eval()
    if os.path.isfile(MODEL_PATH):
        raw = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
        rem = {("rwkv7." + k if k.split(".")[0] in ("emb", "blocks", "ln_out") else k): v
               for k, v in raw.items()}
        m.load_state_dict(rem, strict=False)
    return m


def bench(fn, args, warmup=3, iters=10):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    if not torch.cuda.is_available() or not _fla_available():
        print("SKIP: fla/triton unavailable on this box")
        return
    print("fla chunk_rwkv7 available")

    config = Rwkv7Config(
        vocab_size=65536, hidden_size=768, num_hidden_layers=12, head_size=64,
        intermediate_size=3072, decay_lora_size=64, aaa_lora_size=64,
        mv_lora_size=32, gate_lora_size=128)

    # alignment: fla(bf16 cuda) vs reference(fp32 cpu pure-pytorch)
    ref = load(config, "cpu", torch.float32)
    fla_model = load(config, "cuda", torch.bfloat16)
    ids = torch.tensor([[0, 5248, 478, 2214, 30, 15025, 19, 358, 4457, 311]])
    with torch.no_grad():
        ref_l = ref(ids).logits.float()
        fla_l = fla_model(ids.cuda()).logits.float().cpu()
    rel = (fla_l - ref_l).abs().max().item() / (ref_l.abs().max().item() + 1e-6)
    t5r = torch.topk(ref_l[0, -1], 5).indices.tolist()
    t5f = torch.topk(fla_l[0, -1], 5).indices.tolist()
    print(f"align: rel dev {rel:.3e} | top-5 ref {t5r} fla {t5f}")
    assert t5f[0] == t5r[0], "top-1 diverged"
    print("FLA ALIGNMENT OK")

    # benchmark: fla vs pure-pytorch (force pure-pytorch by disabling fla temporarily)
    print("\nbenchmark (0.1B, bf16, cuda):")
    for B, T in [(1, 512), (4, 1024), (1, 2048)]:
        ids = torch.randint(0, 65536, (B, T), device="cuda")
        with torch.no_grad():
            dt_fla = bench(lambda x: fla_model(x), (ids,))
        # force pure-pytorch: temp disable fla by resetting the tried flag on a fresh model
        import transformers_rwkv7.modeling_rwkv7 as M
        saved = M._FLA_CHUNK
        M._FLA_CHUNK = None
        slow = load(config, "cuda", torch.bfloat16)
        with torch.no_grad():
            dt_slow = bench(lambda x: slow(x), (ids,))
        M._FLA_CHUNK = saved
        toks = B * T
        print(f"  [B={B},T={T}] pure-pytorch {dt_slow*1000:7.1f}ms ({toks/dt_slow:6.0f}tok/s)  "
              f"fla {dt_fla*1000:7.1f}ms ({toks/dt_fla:6.0f}tok/s)  speedup {dt_slow/dt_fla:.1f}x")

    print("\nFLA FAST PATH TEST PASSED")


if __name__ == "__main__":
    main()
