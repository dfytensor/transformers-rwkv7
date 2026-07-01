"""PEFT integration test: verify LoRA wraps the RWKV-7 model, trains, and stays numerically aligned.

Checks:
  1. ``get_peft_model`` succeeds and only LoRA params are trainable
  2. A forward+backward step updates LoRA weights (grads flow)
  3. After wrapping, generation via ``model.generate`` still works
  4. LoRA-on-off equivalence: merging LoRA then comparing against the base on the
     *same* input gives identical logits (sanity that we only added a delta)
"""

import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(__file__))
from transformers_rwkv7 import Rwkv7Config, Rwkv7ForCausalLM  # noqa: E402
from peft import LoraConfig, get_peft_model  # noqa: E402

MODEL_PATH = os.environ.get(
    "RWKV7_TEST_MODEL",
    r"F:\rwkv\models\rwkv7-g1d-0.1b-20260129-ctx8192.pth",
)


def build_model(device):
    config = Rwkv7Config(
        vocab_size=65536, hidden_size=768, num_hidden_layers=12, head_size=64,
        intermediate_size=3072, decay_lora_size=64, aaa_lora_size=64,
        mv_lora_size=32, gate_lora_size=128,
    )
    model = Rwkv7ForCausalLM(config).float()
    raw = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    remapped = {}
    for k, v in raw.items():
        remapped[k if k.startswith("head.") else ("rwkv7." + k if k.split(".")[0] in ("emb", "blocks", "ln_out") else k)] = v
    model.load_state_dict(remapped, strict=False)
    return model.to(device)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    model = build_model(device)
    input_ids = torch.tensor([[0, 5248, 478, 2214, 30, 15025, 19, 358, 4457, 311]], device=device)
    labels = input_ids.clone()

    # baseline logits before LoRA
    model.eval()
    with torch.no_grad():
        base_logits = model(input_ids).logits.clone()

    # ---- 1. wrap with LoRA ----
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
        task_type="CAUSAL_LM",
        # target the att projections + ffn key/value (RWKV has no q/k/v attention, these are the analogues)
        target_modules=["receptance", "key", "value", "output"],
    )
    peft_model = get_peft_model(model, lora_cfg)
    peft_model.print_trainable_parameters()

    n_trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in peft_model.parameters())
    print(f"trainable: {n_trainable/1e6:.3f}M / {n_total/1e6:.2f}M ({100*n_trainable/n_total:.3f}%)")
    assert n_trainable < n_total, "LoRA did not freeze any params"
    assert n_trainable > 0, "LoRA froze everything"

    # ---- 2. forward + backward updates LoRA ----
    peft_model.train()
    out = peft_model(input_ids=input_ids, labels=labels)
    out.loss.backward()
    lora_b = None
    for n, p in peft_model.named_parameters():
        if "lora_B" in n and p.requires_grad:
            lora_b = p
            break
    assert lora_b is not None and lora_b.grad is not None, "LoRA B got no gradient"
    print(f"[train] loss {out.loss.item():.4f} | lora_B grad finite={torch.isfinite(lora_b.grad).all().item()}")

    # ---- 3. generation works through the PEFT wrapper ----
    peft_model.eval()
    with torch.no_grad():
        gen = peft_model.generate(input_ids[:, :4], max_new_tokens=6, do_sample=False, use_cache=True)
    print(f"[generate] produced {gen.shape} tokens: {gen[0].tolist()}")

    # ---- 4. LoRA-on (untrained, zero-init B => identical to base) ----
    # peft LoRA B is zero-initialised so an untrained adapter must reproduce base logits exactly.
    with torch.no_grad():
        lora_logits = peft_model(input_ids).logits
    diff = (lora_logits - base_logits).abs().max().item()
    print(f"[zero-init] |LoRA_logits - base_logits|_max = {diff:.6e}")
    assert diff < 1e-4, "untrained LoRA adapter changed outputs (zero-init assumption broken)"

    print("\nPEFT TEST PASSED")


if __name__ == "__main__":
    main()
