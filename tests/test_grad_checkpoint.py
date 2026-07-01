"""Gradient checkpointing test for RWKV-7.

Verifies:
  1. ``model.gradient_checkpointing_enable()`` works with NO deprecation warning (new format)
  2. Activating it lowers peak activation memory (the whole point)
  3. Gradients are numerically correct vs the non-checkpointed path
"""

import io
import os
import sys
import contextlib

import torch

sys.path.insert(0, os.path.dirname(__file__))
from transformers_rwkv7 import Rwkv7Config, Rwkv7ForCausalLM  # noqa: E402


def _max_alloc(fn):
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1e6
    # CPU: approximate via tracked tensor bytes — not precise, so just return 0 marker
    return 0.0


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {dev}")
    if dev != "cuda":
        print("NOTE: memory comparison needs CUDA; on CPU we only check correctness + no warning")

    # moderately sized so checkpointing memory delta is visible
    config = Rwkv7Config(
        vocab_size=4096, hidden_size=512, num_hidden_layers=8, head_size=64,
        intermediate_size=2048, decay_lora_size=32, aaa_lora_size=32,
        mv_lora_size=32, gate_lora_size=64,
    )
    B, T = 2, 256
    ids = torch.randint(0, config.vocab_size, (B, T), device=dev)
    labels = ids.clone()

    # --- 1. enable() must not emit the deprecation warning ---
    model = Rwkv7ForCausalLM(config).to(dev)
    warn_buf = io.StringIO()
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with contextlib.redirect_stderr(warn_buf):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    dep = [w for w in caught if "old version of the checkpointing format" in str(w.message)]
    print(f"deprecation warnings on enable(): {len(dep)}")
    assert len(dep) == 0, "still using deprecated GC format"
    assert model.is_gradient_checkpointing and model.rwkv7.gradient_checkpointing is True
    print("[1] enable() works, new format, no warning OK")

    # --- 2. memory: checkpointed < non-checkpointed during TRAINING (CUDA only) ---
    if dev == "cuda":
        def train_step_peak(gc_on):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            m = Rwkv7ForCausalLM(config).cuda()
            if gc_on:
                m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            m.train()
            out = m(input_ids=ids, labels=labels)
            out.loss.backward()
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() / 1e6

        mem_off = train_step_peak(gc_on=False)
        del_ids = ids  # keep
        mem_on = train_step_peak(gc_on=True)
        print(f"[2] peak train mem  gc=off {mem_off:.0f}MB  gc=on {mem_on:.0f}MB")
        assert mem_on < mem_off, f"checkpointing did not reduce memory ({mem_on} >= {mem_off})"
        print(f"    saved {mem_off - mem_on:.0f}MB OK")

    # --- 3. gradient correctness: gc-on grads == gc-off grads ---
    torch.manual_seed(0)
    m_off = Rwkv7ForCausalLM(config).to(dev)
    torch.manual_seed(0)
    m_on = Rwkv7ForCausalLM(config).to(dev)
    m_on.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    m_on.train(); m_off.train()

    # clone inputs so backward is independent
    out_off = m_off(input_ids=ids, labels=labels)
    out_off.loss.backward()
    out_on = m_on(input_ids=ids, labels=labels)
    out_on.loss.backward()

    g_off = m_off.rwkv7.blocks[3].att.receptance.weight.grad
    g_on = m_on.rwkv7.blocks[3].att.receptance.weight.grad
    gdiff = (g_off - g_on).abs().max().item()
    ldiff = abs(out_off.loss.item() - out_on.loss.item())
    print(f"[3] loss diff {ldiff:.3e}  grad diff {gdiff:.3e}")
    assert ldiff < 1e-4, "loss diverged between gc on/off"
    assert gdiff < 1e-4, "gradient diverged between gc on/off"
    print("    grads match non-checkpointed path OK")

    print("\nGRADIENT CHECKPOINTING TEST PASSED")


if __name__ == "__main__":
    main()
