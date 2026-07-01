"""Generation throughput benchmark — the bounty's headline metric.

Bounty anchor: RTX 5090, bsz=1, generate >= 145 tok/s.
We measure bsz=1 (and a few batch sizes) autoregressive generation tok/s on the local GPU.
Decode uses the fla Triton fast path (T=1 chunk_rwkv7) + the instance-stashed recurrent state.
"""

import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
import transformers_rwkv7  # noqa: F401
from transformers import AutoModelForCausalLM  # noqa: E402
from transformers_rwkv7 import Rwkv7Tokenizer  # noqa: E402

MODEL = r"F:\rwkv\models\rwkv7-0.1b-hf"
VOCAB = r"F:\rwkv\RWKV-LM\RWKV-v7\rwkv_vocab_v20230424.txt"
PROMPT = "The Eiffel tower is in the city of"


def bench_gen(model, ids, n_new, warmup=1, iters=3):
    for _ in range(warmup):
        model.generate(ids, max_new_tokens=n_new, do_sample=False, use_cache=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        model.generate(ids, max_new_tokens=n_new, do_sample=False, use_cache=True)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    return n_new / dt  # tok/s


def main():
    if not os.path.isdir(MODEL):
        print(f"SKIP: {MODEL} not found")
        return
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(MODEL).to(dev).to(torch.bfloat16 if dev == "cuda" else torch.float32).eval()
    tok = Rwkv7Tokenizer(VOCAB)
    print(f"device: {dev} ({torch.cuda.get_device_name(0) if dev=='cuda' else ''})")
    print(f"model: 0.1B, bf16\n")

    pids = torch.tensor([tok.encode(PROMPT)], device=dev)
    print(f"prompt: {PROMPT!r} ({pids.shape[1]} tokens)\n")

    # bsz=1 decode throughput (the bounty anchor)
    for n in [32, 128, 512]:
        tps = bench_gen(model, pids, n)
        print(f"  bsz=1  generate {n:4d} new tokens -> {tps:8.1f} tok/s")

    # show the generated text for sanity
    out = model.generate(pids, max_new_tokens=40, do_sample=False, use_cache=True)
    print(f"\ngenerated: {tok.decode(out[0].tolist())!r}")

    print(f"\nBounty anchor: RTX 5090 bsz=1 >= 145 tok/s")
    print("(this box is RTX 4090D; a 5090 would be ~1.5-2x faster)")


if __name__ == "__main__":
    main()
