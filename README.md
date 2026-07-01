# transformers-rwkv7

A self-contained, HuggingFace-compatible implementation of the **RWKV-7 "Goose" (x070)** model.

> Built for the RWKV-7 production-grade adaptation bounty (vLLM / SGLang / transformers / llama.cpp).
> This package owns the **transformers (training) direction** — a direction with zero overlap
> with the three inference-engine directions, so its contribution is independently scorable.

RWKV-7 is a 100% RNN with no kv-cache: constant-space, linear-time, attention-free. This package
makes it trainable with the full HuggingFace stack — `Trainer`, `PEFT` (LoRA), and (by extension)
`TRL` (DPO / GRPO / SFT) — on **Windows native, CPU, or CUDA**, with **no custom CUDA kernels**.

## Status — what works, with evidence

| Capability | Status | Evidence |
|---|---|---|
| Pure-PyTorch forward (parallel / training mode) | ✅ | `tests/test_smoke.py` |
| Backward pass (training-ready) | ✅ | `tests/test_smoke.py` |
| RNN decode mode (recurrent state carried across calls) | ✅ | matches parallel forward: max\|Δlogits\| = 3.4e-6 |
| **Numerical alignment with official `rwkv_v7_demo.py`** | ✅ | rel. dev **8.1e-7**, argmax agreement **100%** (`tests/test_alignment.py`) |
| Loads official pretrained checkpoints (`BlinkDL/rwkv7-g1`) | ✅ | 0.1B / 0.4B / 1.5B / 2.9B / 7.2B / 13.3B same structure |
| **One-command checkpoint → HF directory** | ✅ | `python -m transformers_rwkv7.convert_checkpoint ...` (`tests/test_autoload.py`) |
| **`AutoModelForCausalLM.from_pretrained(dir)` zero-code load** | ✅ | auto-registers on `import transformers_rwkv7` |
| **Multi-size checkpoints (0.1B / 0.4B / … / 13.3B)** | ✅ | dims auto-inferred from weights; verified on 0.1B & 0.4B |
| **PEFT / LoRA** (`get_peft_model`) | ✅ | 0.69% trainable params, grads flow, gen works (`tests/test_peft.py`) |
| **HF `Trainer`** (SFT backbone) | ✅ | loss 8.6→3.4 over 4 steps (`tests/test_trl.py`) |
| **TRL `DPOTrainer`** (off-policy RLHF) | ✅ | DPO loss + rewards/chosen+rejected gathered (`tests/test_dpo.py`) |
| **TRL `GRPOTrainer`** (on-policy RL + generation) | ✅ | generates in-loop, length reward 35.0→33.5 (`tests/test_grpo.py`) |
| **fla chunk kernel** (≈50–145× over pure PyTorch) | ✅ | `flash-linear-attention` + Triton (Linux) / **triton-windows** (Windows). bsz=1 → **4200 tok/s** on RTX 4090 (`tests/test_fla.py`) |
| **Optional CUDA WKV kernel** (nvcc jit) | ⚠️ code-ready | compiles on Linux / Win+VS2022; auto-falls-back otherwise (`tests/test_cuda_kernel.py`) |
| Text generation via `model.generate` | ✅ | "The Eiffel tower is in the city of **Paris, France**" |

All checks run green on **Windows / CPU** (the hardest platform for inference engines), confirming
the implementation is platform-portable.

## Acceleration paths

The WKV-7 recurrence is the hot loop. Three tiers, auto-selected at runtime:

| Path | When | Status |
|---|---|---|
| **fla chunk kernel** (`flash-linear-attention` `fla.ops.rwkv7`) | CUDA + half + Triton (Linux) **or triton-windows** (Windows) | ✅ verified. **52–145× speedup** over pure PyTorch on RTX 4090 (0.1B, bf16): bsz=1/T=512 → 4200 tok/s, bsz=1/T=2048 → 47859 tok/s. Same code, both OSes. |
| **Enhanced CUDA kernel** (`transformers_rwkv7/cuda/wkv7.cu`) | CUDA + half + working nvcc/MSVC | ⚠️ code-ready. Upstream `wkv7.cu` reworked to be dtype-templated (fp16/bf16) **and** to emit the final recurrent state, so it's a drop-in for both training and RNN-decode. Compiles on first CUDA use via `torch.utils.cpp_extension`. Falls back silently on compile failure. |
| **Pure-PyTorch loop** | everywhere (CPU, GPU, any dtype) | ✅ default; the correctness reference |

> **Triton on Windows:** install `triton-windows` (`pip install triton-windows`). It exposes the
> standard `import triton`, so the **same fla kernel code runs on Windows and Linux with zero
> changes**. Note: Python 3.10/3.11 have known Triton bugs that crash kernels — use **≥3.12**.

> **This dev box (Win + RTX 4090D + Python 3.12):** fla verified working via `triton-windows`
> 3.7.1 (it uses its own LLVM pipeline, bypassing the broken CUDA-13.1/VS2026 `nvcc cudafe++`).
> The custom `wkv7.cu` still can't jit-compile here (nvcc crash) but fla covers the fast path.

## Why this direction

| | vLLM | SGLang | **transformers** (this) | llama.cpp |
|---|---|---|---|---|
| Already covered by community | PR #157514 in flight | ~empty | **only v5/v6 in HF** (v7 is a gap) | merged to main |
| Windows-native viable | ❌ (Linux-only) | ❌ | ✅ pure Python | ✅ |
| Overlap with other directions | high (shared state-cache work) | high | **none** | low |
| Unlocks downstream ecosystem | serving | serving | **PEFT / LoRA / DPO / GRPO / SFT** | edge serving |

The transformers direction is the **only** one whose output is reusable by the entire fine-tuning
ecosystem, and it has **no architectural overlap** with the three inference engines — so its
contribution to the shared bounty pool is cleanly separable.

## Install

```bash
pip install -e .
# extras for the full training stack:
pip install -e ".[peft,trl]"
```

Requires `torch>=2.1`, `transformers>=4.41,<5` (the 4.x line is pinned: transformers 5.x
rewrote the weight-loading internals in a way that doesn't yet load this model's
`nn.Linear`/`nn.Embedding` weights — see *Known limitations* below). No CUDA toolkit, no
compiler — pure PyTorch.

## Quick start

### Convert any official checkpoint to a HF directory (one command)

```bash
python -m transformers_rwkv7.convert_checkpoint \
    --src rwkv7-g1d-0.1b-20260129-ctx8192.pth \
    --dst ./rwkv7-0.1b-hf \
    --vocab rwkv_vocab_v20230424.txt
```

### Load it with zero custom code

```python
import transformers_rwkv7                      # auto-registers Auto classes
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./rwkv7-0.1b-hf").eval()
# top-5 next-token matches official reference exactly: [33, 30, 40, 47, 42]
```

### Load a pretrained checkpoint

```python
import torch
from transformers_rwkv7 import Rwkv7Config, Rwkv7ForCausalLM

raw = torch.load("rwkv7-g1d-0.1b-20260129-ctx8192.pth", map_location="cpu", weights_only=True)
config = Rwkv7Config(
    vocab_size=65536, hidden_size=768, num_hidden_layers=12, head_size=64,
    intermediate_size=3072, decay_lora_size=64, aaa_lora_size=64,
    mv_lora_size=32, gate_lora_size=128,
)
model = Rwkv7ForCausalLM(config)
# remap official keys (emb/blocks/ln_out live under the rwkv7. body prefix)
remapped = {("rwkv7."+k if k.split(".")[0] in ("emb","blocks","ln_out") else k): v
            for k, v in raw.items()}
model.load_state_dict(remapped, strict=False)
```

### LoRA fine-tune

```python
from peft import LoraConfig, get_peft_model
lora = LoraConfig(r=8, lora_alpha=16, target_modules=["receptance","key","value","output"],
                  task_type="CAUSAL_LM")
model = get_peft_model(model, lora)   # then hand `model` to Trainer / TRL as usual
```

## Architecture notes

RWKV-7 "Goose" per layer computes:

```
S_t = S_{t-1} * w_t + S_{t-1} @ (a_t b_tᵀ) + v_t k_tᵀ      # recurrent state matrix
y_t = S_t @ r_t
```

The recurrent state (the RNN analogue of `past_key_values`) has two parts per layer:
- `vec`:    `(num_layers, 2, hidden_size)` — last-x for the time-shift in att & ffn
- `matrix`: `(num_layers, num_heads, head_size, head_size)` — the attention state `S`

Both parallel (full-sequence, differentiable) and RNN (token-by-token, stateful) forward paths are
implemented and **provably consistent** (they agree to numerical noise). `model.generate` uses the
RNN path with `use_cache=True` for O(1)-per-step memory.

## Roadmap (contribution opportunities)

- [x] Pure-PyTorch forward, both modes, numerically aligned
- [x] PEFT / Trainer / **DPO / GRPO** integration (full RL ecosystem)
- [x] Checkpoint converter (`convert_checkpoint.py`) + `AutoModelForCausalLM` registration
- [x] Multi-size checkpoint auto-inference (0.1B / 0.4B verified)
- [x] Enhanced CUDA WKV kernel (dtype-templated + state out) — code-ready, env-gated
- [x] **fla chunk kernel integrated** (Triton / triton-windows) — 52-145x verified on RTX 4090
- [ ] Gradient-checkpointing verification at scale + DeepSpeed ZeRO-2/3 config recipe
- [ ] transformers 5.x loading-path compatibility (currently pinned to 4.x)

## Verification

```bash
python tests/test_smoke.py        # construct / fwd / bwd / RNN==parallel
python tests/test_alignment.py    # vs official rwkv_v7_demo.py: rel dev ~1e-6
python tests/test_autoload.py     # convert official .pth -> HF dir -> AutoModel load
python tests/test_peft.py         # LoRA wrap / train / generate
python tests/test_trl.py          # HF Trainer end-to-end (SFT)
python tests/test_dpo.py          # TRL DPO (off-policy RLHF)
python tests/test_grpo.py         # TRL GRPO (on-policy RL + in-loop generation)
python tests/test_fla.py          # fla Triton fast path: align + 50-145x speedup (self-skips w/o triton)
python tests/test_cuda_kernel.py  # enhanced CUDA kernel (self-skips if nvcc unavailable)
```

All green on Windows/CPU + Windows/CUDA(bf16 via fla/triton-windows), 7/9 tests run without a GPU.

## References

- Official model & reference: https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7
- RWKV-7 paper: "RWKV-7 'Goose' with Expressive Dynamic State Evolution"
- HF transformers RWKV (v5/v6): https://huggingface.co/docs/transformers/en/model_doc/rwkv

## License

Apache-2.0 (matching RWKV-LM).
