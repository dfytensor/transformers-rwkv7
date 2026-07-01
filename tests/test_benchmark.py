"""Benchmark RWKV-7 forward speed: pure-PyTorch vs torch.compile vs fla (if available).

Measures the dominant WKV recurrent path on CUDA. Run:
    python tests/test_benchmark.py
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from transformers_rwkv7 import Rwkv7Config, Rwkv7ForCausalLM  # noqa: E402


def benchmark(fn, args, warmup=3, iters=10, name=""):
    for _ in range(warmup):
        out = fn(*args)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    dt = (time.perf_counter() - t0) / iters
    tokens = args[0].shape[1] * args[0].shape[0]
    print(f"  {name:28s} {dt*1000:8.2f} ms/fwd   {tokens/dt:10.0f} tok/s")
    return out, dt


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}" + (f" ({torch.cuda.get_device_name(0)})" if dev == "cuda" else ""))

    # 0.1B-sized config
    config = Rwkv7Config(
        vocab_size=65536, hidden_size=768, num_hidden_layers=12, head_size=64,
        intermediate_size=3072, decay_lora_size=64, aaa_lora_size=64,
        mv_lora_size=32, gate_lora_size=128,
    )
    model = Rwkv7ForCausalLM(config).to(dev).to(torch.bfloat16 if dev == "cuda" else torch.float32).eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: 0.1B-class, {n_params/1e6:.1f}M params, dtype={next(model.parameters()).dtype}\n")

    for B, T in [(1, 512), (1, 2048), (4, 1024)]:
        ids = torch.randint(0, config.vocab_size, (B, T), device=dev)
        print(f"[B={B}, T={T}]")
        with torch.no_grad():
            benchmark(lambda x: model(x), (ids,), name="eager (pure pytorch)")

        # torch.compile path (only for the smallest shape to keep compile cost bounded)
        if B == 1 and T == 512 and os.environ.get("RWKV7_BENCH_COMPILE", "1") == "1":
            try:
                compiled = torch.compile(model, mode="reduce-overhead", fullgraph=False)
                with torch.no_grad():
                    benchmark(lambda x: compiled(x), (ids,), name="torch.compile")
            except Exception as e:
                print(f"  torch.compile: SKIP ({type(e).__name__}: {str(e)[:80]})")
        print()


if __name__ == "__main__":
    main()
