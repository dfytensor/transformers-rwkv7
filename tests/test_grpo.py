"""TRL GRPO (Group Relative Policy Optimization) compatibility test for RWKV-7.

GRPO is the on-policy RL method behind DeepSeek-R1. Unlike DPO it generates completions
during training, so this also exercises the model's ``generate`` path inside the training
loop. Verifies the full RL stack runs on RWKV-7 with a synthetic length-based reward.
"""

import os
import sys

import torch
from datasets import Dataset

sys.path.insert(0, os.path.dirname(__file__))
from transformers_rwkv7 import Rwkv7Config, Rwkv7ForCausalLM, Rwkv7Tokenizer  # noqa: E402

try:
    from trl import GRPOConfig, GRPOTrainer
except Exception as e:  # pragma: no cover
    print(f"SKIP: trl GRPO import failed ({e})")
    sys.exit(0)

VOCAB = os.environ.get("RWKV7_VOCAB", r"F:\rwkv\RWKV-LM\RWKV-v7\rwkv_vocab_v20230424.txt")


def main():
    if not os.path.isfile(VOCAB):
        print(f"SKIP: vocab not found at {VOCAB}")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")

    # tiny model so generation is fast
    config = Rwkv7Config(
        vocab_size=65536, hidden_size=128, num_hidden_layers=2, head_size=64,
        intermediate_size=128 * 4, decay_lora_size=32, aaa_lora_size=32,
        mv_lora_size=32, gate_lora_size=32,
    )
    model = Rwkv7ForCausalLM(config).to(device).to(torch.float32)

    tokenizer = Rwkv7Tokenizer(VOCAB)
    tokenizer.pad_token_id = 0
    tokenizer.pad_token = "<pad>"
    tokenizer.eos_token = "<pad>"
    tokenizer.eos_token_id = 0

    # synthetic reward: longer completions score higher (arbitrary, just a real-valued signal)
    def length_reward(completions, **kwargs):
        return [float(len(c)) for c in completions]

    ds = Dataset.from_list([
        {"prompt": "Hello"},
        {"prompt": "The sky is"},
        {"prompt": "1+1="},
        {"prompt": "Cats"},
    ])

    cfg = GRPOConfig(
        output_dir="./_grpo_out",
        max_steps=2,
        per_device_train_batch_size=2,
        num_generations=2,
        max_completion_length=6,
        learning_rate=1e-4,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        bf16=(device == "cuda"),
        use_cpu=(device != "cuda"),
        num_iterations=1,
        beta=0.0,
        scale_rewards=False,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=length_reward,
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    print("starting GRPOTrainer.train() ...")
    result = trainer.train()
    step_losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    rewards = [h.get("reward") for h in trainer.state.log_history if "reward" in h]
    print(f"per-step GRPO losses: {step_losses}")
    print(f"per-step rewards: {rewards}")
    print(f"train metrics: {{'train_loss': {result.metrics.get('train_loss')}}}")

    assert len(step_losses) > 0, "no loss logged"
    assert all(torch.isfinite(torch.tensor(l)) for l in step_losses), "non-finite GRPO loss"
    print("\nTRL GRPO TEST PASSED")


if __name__ == "__main__":
    main()
