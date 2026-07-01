"""Convert an official RWKV-7 ``.pth`` checkpoint into a HuggingFace Hub-ready directory.

Output directory contains:
  - ``config.json``            (Rwkv7Config, with ``auto_map`` for hub discovery)
  - ``model.safetensors``      (remapped weights: emb/blocks/ln_out prefixed with ``rwkv7.``)
  - ``generation_config.json``
  - ``rwkv_vocab_v20230424.txt`` (copied if --vocab given)

The result loads with zero custom code:
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("./rwkv7-0.1b-hf", trust_remote_code=True)

Usage:
    python -m transformers_rwkv7.convert_checkpoint \
        --src rwkv7-g1d-0.1b-20260129-ctx8192.pth \
        --dst ./rwkv7-0.1b-hf \
        --vocab rwkv_vocab_v20230424.txt
"""

import argparse
import os
import shutil

import torch
from safetensors.torch import save_file

from .configuration_rwkv7 import Rwkv7Config
from .modeling_rwkv7 import Rwkv7ForCausalLM


def infer_config(raw: dict) -> Rwkv7Config:
    """Derive the config purely from checkpoint weight shapes."""
    n_embd = raw["emb.weight"].shape[1]
    vocab_size = raw["emb.weight"].shape[0]
    n_layer = 1 + max(int(k.split(".")[1]) for k in raw if k.startswith("blocks."))
    head_size = 64  # RWKV-7 x070 is fixed at head_size=64
    head_key = next(k for k in raw if k.endswith(".att.r_k"))
    num_heads = raw[head_key].shape[0]
    assert num_heads == n_embd // head_size, f"head mismatch {num_heads} vs {n_embd//head_size}"
    return Rwkv7Config(
        vocab_size=vocab_size,
        hidden_size=n_embd,
        num_hidden_layers=n_layer,
        head_size=head_size,
        intermediate_size=raw["blocks.0.ffn.key.weight"].shape[0],
        decay_lora_size=raw["blocks.0.att.w1"].shape[1],
        aaa_lora_size=raw["blocks.0.att.a1"].shape[1],
        mv_lora_size=raw["blocks.0.att.v1"].shape[1],
        gate_lora_size=raw["blocks.0.att.g1"].shape[1],
    )


def remap_keys(raw: dict) -> dict:
    """Official keys -> our keys: body modules (emb/blocks/ln_out) get a ``rwkv7.`` prefix."""
    out = {}
    for k, v in raw.items():
        top = k.split(".")[0]
        if top in ("emb", "blocks", "ln_out"):
            out["rwkv7." + k] = v
        else:  # head.* stays as-is (it's a direct attribute of Rwkv7ForCausalLM)
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="official .pth checkpoint")
    ap.add_argument("--dst", required=True, help="output HF directory")
    ap.add_argument("--vocab", default=None, help="rwkv_vocab_v20230424.txt (copied into --dst if given)")
    ap.add_argument("--dtype", default="auto", choices=["auto", "fp32", "bf16", "fp16"],
                    help="output dtype (auto = keep source dtype)")
    args = ap.parse_args()

    os.makedirs(args.dst, exist_ok=True)
    print(f"loading {args.src} ...")
    raw = torch.load(args.src, map_location="cpu", weights_only=True)
    print(f"  {len(raw)} tensors, {sum(v.numel() for v in raw.values())/1e6:.1f}M params")

    config = infer_config(raw)
    print(f"config: vocab={config.vocab_size} hidden={config.hidden_size} "
          f"layers={config.num_hidden_layers} heads={config.num_attention_heads} "
          f"head={config.head_size} ffn={config.intermediate_size}")

    remapped = remap_keys(raw)

    # dtype
    src_dtype = next(iter(raw.values())).dtype
    if args.dtype == "auto":
        out_dtype = src_dtype
    elif args.dtype == "fp32":
        out_dtype = torch.float32
    elif args.dtype == "bf16":
        out_dtype = torch.bfloat16
    else:
        out_dtype = torch.float16
    if out_dtype != src_dtype:
        remapped = {k: v.to(out_dtype) for k, v in remapped.items()}
        config.torch_dtype = str(out_dtype).replace("torch.", "")
    print(f"  dtype: {src_dtype} -> {out_dtype}")

    # write model.safetensors
    save_file(remapped, os.path.join(args.dst, "model.safetensors"), metadata={"format": "pt"})
    print(f"  wrote model.safetensors ({sum(v.numel()*v.element_size() for v in remapped.values())/1e9:.2f} GB)")

    # write config.json with auto_map so AutoModel can find the classes on the Hub
    config.auto_map = {
        "AutoConfig": "transformers_rwkv7.Rwkv7Config",
        "AutoModelForCausalLM": "transformers_rwkv7.Rwkv7ForCausalLM",
        "AutoTokenizer": "transformers_rwkv7.Rwkv7Tokenizer",
    }
    config.save_pretrained(args.dst)
    # generation config
    Rwkv7ForCausalLM(config).generation_config.save_pretrained(args.dst)
    print(f"  wrote config.json + generation_config.json (auto_map registered)")

    if args.vocab:
        shutil.copyfile(args.vocab, os.path.join(args.dst, "rwkv_vocab_v20230424.txt"))
        print(f"  copied tokenizer vocab")

    print(f"\nDone. Load with:")
    print(f"  from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"  m = AutoModelForCausalLM.from_pretrained({args.dst!r}, trust_remote_code=True)")
    print(f"  t = AutoTokenizer.from_pretrained({args.dst!r}, trust_remote_code=True)")


if __name__ == "__main__":
    main()
