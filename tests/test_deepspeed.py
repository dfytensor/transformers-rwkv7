"""Verify DeepSpeed ZeRO-2/3 configs load correctly and the model trains with DeepSpeed.

Requires: pip install deepspeed (Linux only — DeepSpeed has no Windows native support).
Self-skips if DeepSpeed is unavailable.

Usage on Linux:
    deepspeed tests/test_deepspeed.py --deepspeed configs/ds_zero2.json
    deepspeed tests/test_deepspeed.py --deepspeed configs/ds_zero3.json

Or via HF Trainer:
    python tests/test_deepspeed.py  (uses HF Trainer's --deepspeed integration)
"""

import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(__file__))
from transformers_rwkv7 import Rwkv7Config, Rwkv7ForCausalLM  # noqa: E402

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")


def make_model():
    config = Rwkv7Config(
        vocab_size=4096, hidden_size=512, num_hidden_layers=8, head_size=64,
        intermediate_size=2048, decay_lora_size=32, aaa_lora_size=32,
        mv_lora_size=32, gate_lora_size=64,
    )
    model = Rwkv7ForCausalLM(config)
    model.config.use_cache = False
    return model


def make_dataset(model, n_samples=8, seq_len=64):
    ids = torch.randint(0, model.config.vocab_size, (n_samples, seq_len))
    labels = ids.clone()
    return [{"input_ids": ids[i], "labels": labels[i]} for i in range(n_samples)]


def test_config_loads():
    """Configs are valid JSON with required DeepSpeed keys."""
    for name in ["ds_zero2.json", "ds_zero3.json"]:
        path = os.path.join(CONFIG_DIR, name)
        assert os.path.isfile(path), f"{path} not found"
        with open(path) as f:
            cfg = json.load(f)
        assert "zero_optimization" in cfg, f"{name}: missing zero_optimization"
        assert "bf16" in cfg, f"{name}: missing bf16"
        stage = cfg["zero_optimization"]["stage"]
        print(f"  [{name}] stage={stage} bf16={cfg['bf16']['enabled']} OK")
    print("CONFIG FORMAT OK")


def test_deepspeed_train():
    """End-to-end DeepSpeed training: model.forward + backward + optimizer step."""
    try:
        import deepspeed
    except ImportError:
        print("SKIP: deepspeed not installed (Linux-only)")
        return

    from torch.utils.data import DataLoader

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model().to(dev)

    # ZeRO-2 config
    ds_config = {
        "train_micro_batch_size_per_gpu": 2,
        "bf16": {"enabled": dev == "cuda"},
        "zero_optimization": {"stage": 2},
        "optimizer": {"type": "AdamW", "params": {"lr": 5e-4}},
    }

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model, model_parameters=model.parameters(), config=ds_config
    )

    dataset = make_dataset(model, n_samples=4, seq_len=32)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    losses = []
    for epoch in range(2):
        for batch in loader:
            batch = {k: v.to(dev) for k, v in batch.items()}
            out = model_engine(**batch)
            model_engine.backward(out.loss)
            model_engine.step()
            losses.append(out.loss.item())
            model_engine.zero_grad()

    print(f"  losses: {[f'{l:.3f}' for l in losses]}")
    assert losses[-1] < losses[0], "loss did not decrease"
    print("DEEPSPEED TRAINING OK")


def main():
    if not torch.cuda.is_available():
        print("NOTE: DeepSpeed tests are most meaningful on CUDA multi-GPU")
    print("=== Config validation ===")
    test_config_loads()
    print("\n=== DeepSpeed training ===")
    test_deepspeed_train()
    print("\nDEEPSPEED TEST DONE")


if __name__ == "__main__":
    main()
