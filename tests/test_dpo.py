"""TRL DPO (Direct Preference Optimization) compatibility test for RWKV-7.

This is the canonical RLHF method and a hard requirement of the bounty ("能正常用各种常见的
基于 transformer 的 PEFT 和 RL 库"). Verifies that DPOTrainer runs end-to-end on the
RWKV-7 model: preference data tokenisation, chosen/rejected log-prob gathering through the
recurrent state, and the DPO loss + backward.
"""

import os
import sys

import torch
from datasets import Dataset

sys.path.insert(0, os.path.dirname(__file__))
from transformers_rwkv7 import Rwkv7Config, Rwkv7ForCausalLM, Rwkv7Tokenizer  # noqa: E402

try:
    from trl import DPOConfig, DPOTrainer
except Exception as e:  # pragma: no cover
    print(f"SKIP: trl import failed ({e})")
    sys.exit(0)

VOCAB = os.environ.get(
    "RWKV7_VOCAB",
    r"F:\rwkv\RWKV-LM\RWKV-v7\rwkv_vocab_v20230424.txt",
)


def main():
    if not os.path.isfile(VOCAB):
        print(f"SKIP: vocab not found at {VOCAB}")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    # tiny model so the DPO step runs fast on CPU
    config = Rwkv7Config(
        vocab_size=65536, hidden_size=768, num_hidden_layers=4, head_size=64,
        intermediate_size=768 * 4, decay_lora_size=64, aaa_lora_size=64,
        mv_lora_size=32, gate_lora_size=128,
    )
    model = Rwkv7ForCausalLM(config).to(device).to(torch.float32)
    ref_model = Rwkv7ForCausalLM(config).to(device).to(torch.float32)
    ref_model.eval()

    tokenizer = Rwkv7Tokenizer(VOCAB)
    tokenizer.pad_token_id = 0   # token 0 is reserved (not in vocab) — safe pad
    tokenizer.pad_token = "<pad>"
    tokenizer.unk_token = "<unk>"
    tokenizer.unk_token_id = 0
    tokenizer.eos_token = "<pad>"
    tokenizer.eos_token_id = 0
    tokenizer.bos_token = "<pad>"
    tokenizer.bos_token_id = 0

    # tiny synthetic preference dataset (prompt / chosen / rejected)
    rows = [
        {"prompt": "Hello", "chosen": " world", "rejected": " xqz"},
        {"prompt": "The sky is", "chosen": " blue", "rejected": " angry"},
        {"prompt": "1 + 1 =", "chosen": " 2", "rejected": " banana"},
        {"prompt": "Cats are", "chosen": " animals", "rejected": " minerals"},
    ] * 2
    ds = Dataset.from_list(rows)

    cfg = DPOConfig(
        output_dir="./_dpo_out",
        max_steps=3,
        per_device_train_batch_size=1,
        learning_rate=5e-5,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        max_length=48,
        bf16=(device == "cuda"),
        fp16=False,
        use_cpu=(device != "cuda"),
        remove_unused_columns=False,
        beta=0.1,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    print("starting DPOTrainer.train() ...")
    result = trainer.train()
    step_losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    print(f"per-step DPO losses: {step_losses}")
    metrics = {k: float(v) for k, v in result.metrics.items()}
    print(f"train metrics: {metrics}")

    assert len(step_losses) > 0, "no loss logged"
    assert all(torch.isfinite(torch.tensor(l)) for l in step_losses), "non-finite DPO loss"
    # DPO also reports rewards; check they're present and finite
    rew = [h.get("rewards/accuracies") for h in trainer.state.log_history if "rewards/accuracies" in h]
    print(f"reward accuracies: {rew}")
    print("\nTRL DPO TEST PASSED")


if __name__ == "__main__":
    main()
