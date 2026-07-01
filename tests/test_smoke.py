"""Smoke test: construct a tiny RWKV-7 model and run forward passes (no pretrained weights).

Verifies:
  1. Model constructs from config
  2. Parallel (training) forward produces correct-shaped logits + finite loss
  3. Backward pass works (training-ready)
  4. RNN decode mode (T=1 with carried state) matches the tail of a parallel forward
"""

import sys
import torch

from transformers_rwkv7 import Rwkv7Config, Rwkv7ForCausalLM, Rwkv7State


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    # tiny config
    config = Rwkv7Config(
        vocab_size=512,
        hidden_size=128,        # divisible by head_size 64 -> 2 heads
        num_hidden_layers=4,
        head_size=64,
        intermediate_size=256,
    )
    print(f"config: hidden={config.hidden_size} layers={config.num_hidden_layers} "
          f"heads={config.num_attention_heads} head={config.head_size} "
          f"decay_lora={config.decay_lora_size} gate_lora={config.gate_lora_size}")

    model = Rwkv7ForCausalLM(config).to(device).to(torch.float32)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.2f}M")

    # ---- 1. parallel forward + loss + backward ----
    B, T = 2, 16
    input_ids = torch.randint(0, config.vocab_size, (B, T), device=device)
    labels = input_ids.clone()
    out = model(input_ids=input_ids, labels=labels)
    print(f"[parallel] logits {tuple(out.logits.shape)} loss {out.loss.item():.4f}")
    assert out.logits.shape == (B, T, config.vocab_size)
    assert torch.isfinite(out.loss), "loss not finite"

    out.loss.backward()
    g = model.rwkv7.blocks[0].att.w1.grad
    print(f"[parallel] backward OK, w1.grad finite={torch.isfinite(g).all().item()}")

    # ---- 2. RNN decode mode must match the parallel forward tail ----
    model.eval()
    state = None
    logits_seq = []
    with torch.no_grad():
        for t in range(T):
            tok = input_ids[:, t:t + 1]
            o = model(input_ids=tok, past_key_values=state, use_cache=True)
            state = o.past_key_values
            logits_seq.append(o.logits)
    rnn_logits = torch.cat(logits_seq, dim=1)  # (B, T, V)
    print(f"[rnn] logits {tuple(rnn_logits.shape)} state.vec {tuple(state.vec.shape)} state.matrix {tuple(state.matrix.shape)}")

    with torch.no_grad():
        par_logits = model(input_ids=input_ids).logits
    max_diff = (rnn_logits - par_logits).abs().max().item()
    print(f"[match] rnn-vs-parallel max|Δlogits| = {max_diff:.6e}")
    assert max_diff < 1e-3, "RNN and parallel forwards diverged!"

    # ---- 3. state shape sanity ----
    L, H, N, C = config.num_hidden_layers, config.num_attention_heads, config.head_size, config.hidden_size
    assert state.vec.shape == (B, L, 2, C)
    assert state.matrix.shape == (B, L, H, N, N)
    print("[state] shapes OK")

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    sys.exit(main())
