"""Generate text with RWKV-7 "Goose" using the HF-compatible implementation.

Usage:
    python examples/generate.py --model F:/rwkv/models/rwkv7-g1d-0.1b-20260129-ctx8192.pth \
        --vocab F:/rwkv/RWKV-LM/RWKV-v7/rwkv_vocab_v20230424.txt \
        --prompt "The Eiffel tower is in the city of" --max-new-tokens 32
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from transformers_rwkv7 import Rwkv7Config, Rwkv7ForCausalLM, Rwkv7Tokenizer


def load_model(model_path, device):
    raw = torch.load(model_path, map_location="cpu", weights_only=True)
    # infer dims from the checkpoint
    n_embd = raw["emb.weight"].shape[1]
    vocab_size = raw["emb.weight"].shape[0]
    # count layers
    n_layer = 1 + max(int(k.split(".")[1]) for k in raw if k.startswith("blocks."))
    head_size = 64
    config = Rwkv7Config(
        vocab_size=vocab_size, hidden_size=n_embd, num_hidden_layers=n_layer,
        head_size=head_size, intermediate_size=n_embd * 4,
        decay_lora_size=raw["blocks.0.att.w1"].shape[1],
        aaa_lora_size=raw["blocks.0.att.a1"].shape[1],
        mv_lora_size=raw["blocks.0.att.v1"].shape[1],
        gate_lora_size=raw["blocks.0.att.g1"].shape[1],
    )
    model = Rwkv7ForCausalLM(config).float().eval()
    remapped = {}
    for k, v in raw.items():
        remapped[k if k.startswith("head.") else (
            "rwkv7." + k if k.split(".")[0] in ("emb", "blocks", "ln_out") else k)] = v
    model.load_state_dict(remapped, strict=False)
    return model.to(device), config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--prompt", default="The Eiffel tower is in the city of")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--use-cache", action="store_true", default=True,
                    help="use RNN-mode state caching for fast autoregressive generation")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, config = load_model(args.model, device)
    tok = Rwkv7Tokenizer(args.vocab)

    ids = torch.tensor([tok.encode(args.prompt)], device=device)
    print(f"prompt: {args.prompt}")
    print(f"({len(ids[0])} tokens)\n---")

    out = model.generate(
        ids,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        use_cache=args.use_cache,
    )
    text = tok.decode(out[0].tolist())
    print(text)


if __name__ == "__main__":
    main()
