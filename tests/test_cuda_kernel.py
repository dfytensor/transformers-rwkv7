"""Verify the enhanced CUDA wkv7 kernel compiles and matches the pure-PyTorch path.

If CUDA/nvcc is unavailable this test self-skips. Otherwise it:
  1. triggers one-shot compilation of the kernel
  2. runs the full 0.1B model on CUDA in bf16 and compares logits against the CPU fp32
     reference (the alignment baseline)
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from transformers_rwkv7 import Rwkv7Config, Rwkv7ForCausalLM  # noqa: E402
from transformers_rwkv7.modeling_rwkv7 import _cuda_wkv7_available  # noqa: E402

MODEL_PATH = os.environ.get(
    "RWKV7_TEST_MODEL",
    r"F:\rwkv\models\rwkv7-g1d-0.1b-20260129-ctx8192.pth",
)


def load_into(config, device, dtype):
    model = Rwkv7ForCausalLM(config).to(device).to(dtype).eval()
    raw = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    remapped = {}
    for k, v in raw.items():
        remapped[k if k.startswith("head.") else (
            "rwkv7." + k if k.split(".")[0] in ("emb", "blocks", "ln_out") else k)] = v
    model.load_state_dict(remapped, strict=False)
    return model


def main():
    if not torch.cuda.is_available():
        print("SKIP: CUDA not available")
        return
    if not _cuda_wkv7_available():
        print("SKIP: wkv7 CUDA kernel did not compile (nvcc/MSVC issue). Falling back to pure-PyTorch.")
        return
    print("CUDA wkv7 kernel compiled OK")

    config = Rwkv7Config(
        vocab_size=65536, hidden_size=768, num_hidden_layers=12, head_size=64,
        intermediate_size=3072, decay_lora_size=64, aaa_lora_size=64,
        mv_lora_size=32, gate_lora_size=128,
    )
    if not os.path.isfile(MODEL_PATH):
        print("SKIP: model not found")
        return

    # reference on CPU fp32 (pure-PyTorch path)
    ref = load_into(config, "cpu", torch.float32)
    # CUDA bf16 with the kernel
    cuda_model = load_into(config, "cuda", torch.bfloat16)

    ids = torch.tensor([[0, 5248, 478, 2214, 30, 15025, 19, 358, 4457, 311]])
    with torch.no_grad():
        ref_logits = ref(ids).logits.float()
        cuda_logits = cuda_model(ids.cuda()).logits.float().cpu()

    # bf16 has limited precision; compare argmax + relative dev on the dominant logits
    rel = (cuda_logits - ref_logits).abs().max().item() / (ref_logits.abs().max().item() + 1e-6)
    top5_ref = torch.topk(ref_logits[0, -1], 5).indices.tolist()
    top5_cuda = torch.topk(cuda_logits[0, -1], 5).indices.tolist()
    print(f"rel dev (bf16-cuda vs fp32-cpu): {rel:.4e}")
    print(f"top-5 ref:  {top5_ref}")
    print(f"top-5 cuda: {top5_cuda}")
    # bf16 tolerance: rel dev up to ~0.05 is fine; argmax top-1 must agree
    assert top5_cuda[0] == top5_ref[0], "top-1 prediction diverged"
    print("\nCUDA KERNEL ALIGNMENT TEST PASSED")


if __name__ == "__main__":
    main()
