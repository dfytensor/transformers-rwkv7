"""PyTorch RWKV-7 "Goose" (x070) model �?a self-contained HuggingFace-compatible implementation.

Pure-PyTorch reference, no custom CUDA kernels required. Matches the official
``rwkv_v7_demo.py`` forward pass numerically and supports:

  * **Parallel / training mode** �?full sequence forward (B, T, C), differentiable.
  * **RNN / decode mode** �?token-by-token generation carrying a recurrent state.
  * HF ``PreTrainedModel`` conventions so ``Trainer`` / ``PEFT`` / ``TRL`` plug in.

Reference: https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_demo.py
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

from .configuration_rwkv7 import Rwkv7Config


# ----------------------------------------------------------------------------------------------------------------------
# Recurrent state
# ----------------------------------------------------------------------------------------------------------------------

@dataclass
class Rwkv7State:
    """The RWKV-7 recurrent state (replaces the transformer ``past_key_values``).

    For every layer we keep:
      * ``vec``   : (num_layers, 2, hidden_size) �?``[att_last_x, ffn_last_x]``
      * ``matrix``: (num_layers, num_heads, head_size, head_size) �?the attention state S
    A leading batch dim is added when materialised for a forward pass.
    """

    vec: torch.Tensor
    matrix: torch.Tensor

    @classmethod
    def zeros(cls, batch_size: int, config: Rwkv7Config, device=None, dtype=torch.float32) -> "Rwkv7State":
        L, H, N, C = config.num_hidden_layers, config.num_attention_heads, config.head_size, config.hidden_size
        return cls(
            vec=torch.zeros(batch_size, L, 2, C, device=device, dtype=dtype),
            matrix=torch.zeros(batch_size, L, H, N, N, device=device, dtype=dtype),
        )

    def to(self, *args, **kwargs) -> "Rwkv7State":
        return Rwkv7State(self.vec.to(*args, **kwargs), self.matrix.to(*args, **kwargs))

    def detach(self) -> "Rwkv7State":
        return Rwkv7State(self.vec.detach(), self.matrix.detach())


# ----------------------------------------------------------------------------------------------------------------------
# Core WKV-7 recurrent operator (pure PyTorch, matches demo.py:170-203)
#   S_t = S_{t-1} * w_t + S_{t-1} @ (a_t b_t^T) + v_t k_t^T
#   y_t = S_t @ r_t
# ----------------------------------------------------------------------------------------------------------------------

def rwkv7_wkv_headed(
    r: torch.Tensor,  # (B, T, H, N)
    w_pre: torch.Tensor,  # (B, T, H, N)  pre-transform decay
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    state: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Headed WKV-7. All inputs float32 (B, T, H, N). Returns out (B, T, H, N) and state (B, H, N, N)."""
    B, T, H, N = r.shape
    r = r.float()
    k = k.float()
    v = v.float()
    a = a.float()
    b = b.float()
    w = torch.exp(-torch.exp(w_pre.float()))  # decay in [0, 1]

    if state is None:
        state = torch.zeros(B, H, N, N, device=r.device, dtype=torch.float)
    else:
        state = state.float()

    outs = torch.empty(B, T, H, N, device=r.device, dtype=torch.float)
    for t in range(T):
        kt = k[:, t].view(B, H, 1, N)      # (B,H,1,N)
        rt = r[:, t].view(B, H, N, 1)      # (B,H,N,1)
        vt = v[:, t].view(B, H, N, 1)      # (B,H,N,1)
        at = a[:, t].view(B, H, N, 1)      # (B,H,N,1)
        bt = b[:, t].view(B, H, 1, N)      # (B,H,1,N)
        wt = w[:, t].view(B, H, 1, N)      # broadcast over rows
        # S = S * w(row-wise) + S @ (a b^T) + v k^T
        state = state * wt + state @ (at @ bt) + vt @ kt
        outs[:, t] = (state @ rt).view(B, H, N)
    return outs, state


# ----------------------------------------------------------------------------------------------------------------------
# Optional fast CUDA path — compiled lazily on first CUDA use via torch cpp_extension.
# Drops the pure-PyTorch Python loop (the dominant cost) for ~10x+ speedup. Falls back
# transparently to the pure-PyTorch loop on CPU / when nvcc is unavailable.
# ----------------------------------------------------------------------------------------------------------------------

_CUDA_WKV7 = None  # cached compiled op handle
_CUDA_WKV7_TRIED = False  # one-shot compile attempt flag


def _cuda_wkv7_available() -> bool:
    """True iff the enhanced wkv7 CUDA kernel compiled successfully (tried at most once)."""
    global _CUDA_WKV7, _CUDA_WKV7_TRIED
    if _CUDA_WKV7 is not None:
        return True
    if _CUDA_WKV7_TRIED or not torch.cuda.is_available():
        return False
    _CUDA_WKV7_TRIED = True
    try:
        _load_cuda_wkv7()
        return _CUDA_WKV7 is not None
    except Exception:
        return False


def _load_cuda_wkv7():
    """Lazy-compile the enhanced wkv7 CUDA kernel (fp16/bf16 + state out)."""
    global _CUDA_WKV7
    if _CUDA_WKV7 is not None:
        return _CUDA_WKV7
    import os
    from torch.utils.cpp_extension import load
    here = os.path.dirname(os.path.abspath(__file__))
    cuda_dir = os.path.join(here, "cuda")
    # Minimal flag set for broad toolchain compatibility (older nvcc + newest MSVC combos
    # can crash on the more aggressive optimisation flags). -D_N_ sets head_size=64.
    flags = ["-O3", "-allow-unsupported-compiler", f"-D_N_=64"]
    load(name="rwkv7_cuda",
         sources=[os.path.join(cuda_dir, "wkv7_op.cpp"), os.path.join(cuda_dir, "wkv7.cu")],
         is_python_module=False, verbose=False, extra_cuda_cflags=flags)
    _CUDA_WKV7 = torch.ops.rwkv7_cuda
    return _CUDA_WKV7


def _wkv7_cuda(r, w_pre, k, v, a, b, state):
    """Call the compiled kernel. Inputs (B,T,H,N) in fp16/bf16; returns (out (B,T,H,N), state (B,H,N,N) fp32)."""
    op = _load_cuda_wkv7()
    B, T, H, N = r.shape
    # kernel expects (B,T,C=H*N) contiguous in fp16/bf16
    r2 = r.reshape(B, T, H * N).contiguous()
    w2 = w_pre.reshape(B, T, H * N).contiguous()
    k2 = k.reshape(B, T, H * N).contiguous()
    v2 = v.reshape(B, T, H * N).contiguous()
    a2 = a.reshape(B, T, H * N).contiguous()
    b2 = b.reshape(B, T, H * N).contiguous()
    st = state.contiguous() if state is not None else torch.empty(0, device=r.device)
    y, new_state = op.forward(B, T, H * N, H, r2, w2, k2, v2, a2, b2, st)
    return y.view(B, T, H, N), new_state


# ----------------------------------------------------------------------------------------------------------------------
# Optional fast path: fla (flash-linear-attention) Triton chunk kernel.
# Works on Linux (triton) and Windows (triton-windows); auto-detected. This is the
# fastest tier — chunked parallel WKV, ~10-30x over the pure-PyTorch loop on training.
# ----------------------------------------------------------------------------------------------------------------------

_FLA_CHUNK = None
_FLA_TRIED = False


def _fla_available() -> bool:
    global _FLA_CHUNK, _FLA_TRIED
    if _FLA_CHUNK is not None:
        return True
    if _FLA_TRIED or not torch.cuda.is_available():
        return False
    _FLA_TRIED = True
    try:
        from fla.ops.rwkv7 import chunk_rwkv7  # noqa: F401
        _FLA_CHUNK = chunk_rwkv7
        return True
    except Exception:
        return False


def _wkv7_fla(r, w_pre, k, v, a, b, state):
    """fla Triton chunk kernel. Inputs (B,T,H,N) half-precision; returns (out (B,T,H,N), state (B,H,N,N) fp32).

    Convention: fla's ``w`` is the *log decay* (the kernel computes ``decay = exp(w)``).
    Our pre-decay value ``w_pre`` yields ``decay = exp(-exp(w_pre))``, so we pass ``w = -exp(w_pre)``.
    All inputs are coerced to a single dtype (fla's dot op requires uniform dtype; under
    autocast the time-mix sub-tensors can be a bf16/fp32 mix).
    """
    dt = r.dtype
    w_log = (-torch.exp(w_pre.float())).to(dt)  # log(decay) = -exp(w_pre)
    out, new_state = _FLA_CHUNK(
        r=r.to(dt).contiguous(), w=w_log.contiguous(),
        k=k.to(dt).contiguous(), v=v.to(dt).contiguous(),
        a=a.to(dt).contiguous(), b=b.to(dt).contiguous(),
        scale=1.0,
        initial_state=state.to(dt).contiguous() if state is not None else None,
        output_final_state=True,
    )
    return out, new_state


# ----------------------------------------------------------------------------------------------------------------------
# Layers
# ----------------------------------------------------------------------------------------------------------------------

class Rwkv7TimeMix(nn.Module):
    """RWKV-7 x070 time-mixing (attention-equivalent) block �?pure PyTorch."""

    def __init__(self, config: Rwkv7Config, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_size = config.head_size
        H, N, C = self.num_heads, self.head_size, self.hidden_size

        # time-shift mixing coefficients (per-channel)
        self.x_r = nn.Parameter(torch.empty(1, 1, C))
        self.x_w = nn.Parameter(torch.empty(1, 1, C))
        self.x_k = nn.Parameter(torch.empty(1, 1, C))
        self.x_v = nn.Parameter(torch.empty(1, 1, C))
        self.x_a = nn.Parameter(torch.empty(1, 1, C))
        self.x_g = nn.Parameter(torch.empty(1, 1, C))

        # decay LoRA (w)
        self.w0 = nn.Parameter(torch.empty(1, 1, C))
        self.w1 = nn.Parameter(torch.empty(C, config.decay_lora_size))
        self.w2 = nn.Parameter(torch.empty(config.decay_lora_size, C))

        # in-context learning rate gate (a)
        self.a0 = nn.Parameter(torch.empty(1, 1, C))
        self.a1 = nn.Parameter(torch.empty(C, config.aaa_lora_size))
        self.a2 = nn.Parameter(torch.empty(config.aaa_lora_size, C))

        # value residual (v)
        self.v0 = nn.Parameter(torch.empty(1, 1, C))
        self.v1 = nn.Parameter(torch.empty(C, config.mv_lora_size))
        self.v2 = nn.Parameter(torch.empty(config.mv_lora_size, C))

        # output gate (g)
        self.g1 = nn.Parameter(torch.empty(C, config.gate_lora_size))
        self.g2 = nn.Parameter(torch.empty(config.gate_lora_size, C))

        # per-head modulation
        self.k_k = nn.Parameter(torch.empty(1, 1, C))
        self.k_a = nn.Parameter(torch.empty(1, 1, C))
        self.r_k = nn.Parameter(torch.empty(H, N))

        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        # note: eps=64e-5 (NOT the usual 1e-5) — matches official x070
        self.ln_x = nn.GroupNorm(H, C, eps=config.group_norm_eps)
        # keep config reference for HF-style _init_weights dispatch (called by post_init/init_weights)
        self._config = config

    def _apply_init(self):
        """RWKV-7 x070 layer-wise weight init. Invoked via Rwkv7PreTrainedModel._init_weights.

        Guarded against meta tensors so it is a no-op during ``from_pretrained``'s
        empty-weights construction phase (the real values are then loaded from the checkpoint).
        """
        if self.x_r.is_meta:
            return
        layer_id = self.layer_id
        config = self._config
        C = self.hidden_size
        ratio_0_to_1 = layer_id / max(1, config.num_hidden_layers - 1)
        ratio_1_to_almost0 = 1.0 - (layer_id / config.num_hidden_layers)
        ddd = torch.arange(C, dtype=torch.float32) / C

        with torch.no_grad():
            self.x_r.data = (1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0)).view(1, 1, C)
            self.x_w.data = (1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0)).view(1, 1, C)
            self.x_k.data = (1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0)).view(1, 1, C)
            self.x_v.data = (1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0)).view(1, 1, C)
            self.x_a.data = (1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0)).view(1, 1, C)
            self.x_g.data = (1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0)).view(1, 1, C)

            N = self.head_size
            linear = torch.arange(C, dtype=torch.float32) / (C - 1) - 0.5
            zigzag = ((torch.arange(C) % N) - ((N - 1) / 2)) / ((N - 1) / 2)
            zigzag = zigzag * zigzag.abs()
            www = -6.0 + 6.0 * (torch.arange(C, dtype=torch.float32) / (C - 1)) ** (1 + 1 * ratio_0_to_1 ** 0.3)

            self.w0.data = (www.view(1, 1, C) + 0.5 + zigzag * 2.5)
            self.a0.data = torch.zeros(1, 1, C) - 0.19 + zigzag * 0.3 + linear * 0.4
            self.v0.data = torch.zeros(1, 1, C) + 0.73 - linear * 0.4
            self.k_k.data = torch.zeros(1, 1, C) + 0.71 - linear * 0.1
            self.k_a.data = torch.zeros(1, 1, C) + 1.02
            self.r_k.data = torch.zeros(self.num_heads, N) - 0.04

            nn.init.orthogonal_(self.w2, gain=0.1 * (max(1, (C / self.w2.shape[0])) ** 0.5))
            nn.init.orthogonal_(self.a2, gain=0.1 * (max(1, (C / self.a2.shape[0])) ** 0.5))
            nn.init.orthogonal_(self.v2, gain=0.1 * (max(1, (C / self.v2.shape[0])) ** 0.5))
            nn.init.orthogonal_(self.g2, gain=0.1 * (max(1, (C / self.g2.shape[0])) ** 0.5))
            nn.init.zeros_(self.w1)
            nn.init.zeros_(self.a1)
            nn.init.zeros_(self.v1)
            nn.init.zeros_(self.g1)

            self.receptance.weight.data.uniform_(-0.5 / (C ** 0.5), 0.5 / (C ** 0.5))
            self.key.weight.data.uniform_(-0.05 / (C ** 0.5), 0.05 / (C ** 0.5))
            self.value.weight.data.uniform_(-0.5 / (C ** 0.5), 0.5 / (C ** 0.5))
            self.output.weight.data.zero_()

    def forward(
        self,
        x: torch.Tensor,          # (B, T, C) already layernorm'd input
        v_first: torch.Tensor,    # (B, T, C)
        last_x: Optional[torch.Tensor],  # (B, C) previous-token x for RNN mode; None => GPT shift
        matrix_state: Optional[torch.Tensor],  # (B, H, N, N) recurrent attention state
        compute_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (out, v_first_updated, new_last_x, new_matrix_state)."""
        B, T, C = x.shape
        H, N = self.num_heads, self.head_size

        # --- time shift ---
        if last_x is not None and T == 1:
            # RNN decode mode: shift against the previous token's x
            xx = last_x.unsqueeze(1) - x  # (B,1,C)
        else:
            # parallel mode: shift along time using zero-pad
            xx = F.pad(x, (0, 0, 1, -1)) - x  # previous-position minus current
        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        # decay: soft-clamp to (-inf, -0.5); exp(-exp(w)) happens inside the WKV kernel
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v
        else:
            v = v + (v_first - v) * torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, N), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)

        # WKV-7: the "a" arg of the operator is -kk, the "b" arg is kk*a (see demo.py)
        # Fast path tiers (auto-selected): fla Triton chunk > enhanced CUDA kernel > pure-PyTorch
        use_half = r.dtype in (torch.float16, torch.bfloat16)
        use_fla = (r.is_cuda and use_half and _fla_available())
        use_cuda = (not use_fla and r.is_cuda and use_half and _cuda_wkv7_available())
        a_h = (-kk).view(B, T, H, N)
        b_h = (kk * a).view(B, T, H, N)
        if use_fla:
            out_h, new_state = _wkv7_fla(
                r.view(B, T, H, N), w.view(B, T, H, N), k.view(B, T, H, N),
                v.view(B, T, H, N), a_h, b_h, matrix_state)
            out = out_h.view(B, T, C).to(x.dtype)
        elif use_cuda:
            out_h, new_state = _wkv7_cuda(
                r.view(B, T, H, N), w.view(B, T, H, N), k.view(B, T, H, N),
                v.view(B, T, H, N), a_h, b_h, matrix_state)
            out = out_h.view(B, T, C).to(x.dtype)
        else:
            r_h = r.view(B, T, H, N).to(compute_dtype)
            w_h = w.view(B, T, H, N).to(compute_dtype)
            k_h = k.view(B, T, H, N).to(compute_dtype)
            v_h = v.view(B, T, H, N).to(compute_dtype)
            out_h, new_state = rwkv7_wkv_headed(
                r_h, w_h, k_h, v_h, a_h.to(compute_dtype), b_h.to(compute_dtype),
                state=matrix_state)
            out = out_h.view(B, T, C).to(x.dtype)

        out = self.ln_x(out.view(B * T, C)).view(B, T, C)
        out = out + (
            (r.view(B, T, H, N) * k.view(B, T, H, N) * self.r_k).sum(dim=-1, keepdim=True)
            * v.view(B, T, H, N)
        ).view(B, T, C)
        out = self.output(out * g)

        new_last_x = x[:, -1]
        return out, v_first, new_last_x, new_state


class Rwkv7ChannelMix(nn.Module):
    """RWKV-7 x070 channel-mixing (FFN-equivalent) block �?pure PyTorch."""

    def __init__(self, config: Rwkv7Config, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        C = config.hidden_size
        self.x_k = nn.Parameter(torch.empty(1, 1, C))
        self.key = nn.Linear(C, config.intermediate_size, bias=False)
        self.value = nn.Linear(config.intermediate_size, C, bias=False)
        self._config = config

    def _apply_init(self):
        if self.x_k.is_meta:
            return
        layer_id = self.layer_id
        config = self._config
        C = config.hidden_size
        ratio_1_to_almost0 = 1.0 - (layer_id / config.num_hidden_layers)
        ddd = torch.arange(C, dtype=torch.float32) / C
        with torch.no_grad():
            self.x_k.data = (1.0 - torch.pow(ddd, ratio_1_to_almost0 ** 4)).view(1, 1, C)
        self.key.weight.data.uniform_(-0.5 / (C ** 0.5), 0.5 / (C ** 0.5))
        self.value.weight.data.zero_()

    def forward(
        self,
        x: torch.Tensor,         # (B, T, C)
        last_x: Optional[torch.Tensor],  # (B, C) for RNN mode
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if last_x is not None and x.size(1) == 1:
            xx = last_x.unsqueeze(1) - x
        else:
            xx = F.pad(x, (0, 0, 1, -1)) - x
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        out = self.value(k)
        return out, x[:, -1]


class Rwkv7Block(nn.Module):
    def __init__(self, config: Rwkv7Config, layer_id: int):
        super().__init__()
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        if layer_id == 0:
            self.ln0 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.ln0 = None
        self.att = Rwkv7TimeMix(config, layer_id)
        self.ffn = Rwkv7ChannelMix(config, layer_id)

    def forward(
        self,
        x: torch.Tensor,
        v_first: torch.Tensor,
        att_last_x: Optional[torch.Tensor],
        ffn_last_x: Optional[torch.Tensor],
        matrix_state: Optional[torch.Tensor],
        compute_dtype: torch.dtype,
    ):
        if self.ln0 is not None:
            x = self.ln0(x)
        h = self.ln1(x)
        attn, v_first, new_att_last, new_matrix = self.att(h, v_first, att_last_x, matrix_state, compute_dtype)
        x = x + attn
        h = self.ln2(x)
        ffn_out, new_ffn_last = self.ffn(h, ffn_last_x)
        x = x + ffn_out
        return x, v_first, new_att_last, new_ffn_last, new_matrix


# ----------------------------------------------------------------------------------------------------------------------
# Pre-trained base
# ----------------------------------------------------------------------------------------------------------------------

class Rwkv7PreTrainedModel(PreTrainedModel):
    config_class = Rwkv7Config
    base_model_prefix = "rwkv7"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Rwkv7Block"]
    _skip_keys_device_placement = ["past_key_values"]
    _keep_in_fp32_modules = []

    def _init_weights(self, module):
        # RWKV-specific per-layer init (time-mix / channel-mix) — uses layer_id from the module.
        # Guarded against meta tensors inside _apply_init, so safe under from_pretrained's empty construction.
        if isinstance(module, Rwkv7TimeMix):
            module._apply_init()
            return
        if isinstance(module, Rwkv7ChannelMix):
            module._apply_init()
            return
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.hidden_size ** -0.5)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.hidden_size ** -0.5)
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (Rwkv7Model, Rwkv7ForCausalLM)):
            module.gradient_checkpointing = value


# ----------------------------------------------------------------------------------------------------------------------
# Model body
# ----------------------------------------------------------------------------------------------------------------------

class Rwkv7Model(Rwkv7PreTrainedModel):
    def __init__(self, config: Rwkv7Config):
        super().__init__(config)
        self.emb = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([Rwkv7Block(config, i) for i in range(config.num_hidden_layers)])
        self.ln_out = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.head_size = config.head_size
        self.num_heads = config.num_attention_heads
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[Rwkv7State] = None,
        use_cache: Optional[bool] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # transformers >=4.41 may pass a Cache/DynamicCache from generate(); RWKV uses its own
        # Rwkv7State. Coerce any foreign cache object to None (fresh state).
        if past_key_values is not None and not isinstance(past_key_values, Rwkv7State):
            past_key_values = None

        if inputs_embeds is None:
            inputs_embeds = self.emb(input_ids)

        B, T, C = inputs_embeds.shape
        L = len(self.blocks)
        compute_dtype = torch.float32  # the WKV loop is numerically safe in fp32

        # per-layer "previous x" for the time-shift. In RNN mode these come from state.
        if past_key_values is not None:
            att_last = past_key_values.vec[:, :, 0]   # (B, L, C)
            ffn_last = past_key_values.vec[:, :, 1]   # (B, L, C)
            matrices = past_key_values.matrix         # (B, L, H, N, N)
        else:
            att_last = [None] * L
            ffn_last = [None] * L
            matrices = [None] * L

        x = inputs_embeds
        v_first = torch.zeros(B, T, C, device=x.device, dtype=x.dtype)

        new_att_last = [None] * L
        new_ffn_last = [None] * L
        new_matrices = [None] * L

        for i, block in enumerate(self.blocks):
            cur_att_last = att_last[:, i] if past_key_values is not None else None
            cur_ffn_last = ffn_last[:, i] if past_key_values is not None else None
            cur_matrix = matrices[:, i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:
                x, v_first, na, nf, nm = self._gc_block(block, x, v_first, cur_att_last, cur_ffn_last, cur_matrix, compute_dtype)
            else:
                x, v_first, na, nf, nm = block(x, v_first, cur_att_last, cur_ffn_last, cur_matrix, compute_dtype)
            new_att_last[i] = na
            new_ffn_last[i] = nf
            new_matrices[i] = nm

        x = self.ln_out(x)

        next_state = None
        if use_cache:
            vec = torch.stack(
                [torch.stack([new_att_last[i], new_ffn_last[i]], dim=1) for i in range(L)], dim=1
            )  # (B, L, 2, C)
            matrix = torch.stack(new_matrices, dim=1)  # (B, L, H, N, N)
            next_state = Rwkv7State(vec=vec, matrix=matrix)

        return BaseModelOutputWithPast(last_hidden_state=x, past_key_values=next_state, hidden_states=None, attentions=None)

    def _gc_block(self, block, x, v_first, att_last, ffn_last, matrix, compute_dtype):
        def custom(*args):
            x, v_first, att_last, ffn_last, matrix = args
            return block(x, v_first, att_last, ffn_last, matrix, compute_dtype)
        return torch.utils.checkpoint.checkpoint(custom, x, v_first, att_last, ffn_last, matrix, use_reentrant=False)


# ----------------------------------------------------------------------------------------------------------------------
# Causal LM head
# ----------------------------------------------------------------------------------------------------------------------

class Rwkv7ForCausalLM(Rwkv7PreTrainedModel, GenerationMixin):
    _tied_weights_keys = []

    def __init__(self, config: Rwkv7Config):
        super().__init__(config)
        self.rwkv7 = Rwkv7Model(config)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.rwkv7.emb

    def set_input_embeddings(self, value):
        self.rwkv7.emb = value

    def get_output_embeddings(self):
        return self.head

    def set_output_embeddings(self, new):
        self.head = new

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Rwkv7State] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs = self.rwkv7(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            attention_mask=attention_mask,
        )
        hidden = outputs.last_hidden_state
        logits = self.head(hidden)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            if attention_mask is not None:
                shift_mask = attention_mask[..., :-1].contiguous()
                loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)
                loss = (loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                                 shift_labels.view(-1)).view(-1) * shift_mask.view(-1)).sum() / shift_mask.sum().clamp(min=1)
            else:
                loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutputWithPast(
            loss=loss, logits=logits, past_key_values=outputs.past_key_values,
            hidden_states=None, attentions=None,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, attention_mask=None, **kwargs):
        # Standard HF generation hook. In decode mode we feed one token at a time and reuse state.
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": True,
            "attention_mask": attention_mask,
        }

    def _reorder_cache(self, past_key_values: Rwkv7State, beam_idx: torch.Tensor) -> Rwkv7State:
        return Rwkv7State(
            vec=past_key_values.vec.index_select(0, beam_idx.to(past_key_values.vec.device)),
            matrix=past_key_values.matrix.index_select(0, beam_idx.to(past_key_values.matrix.device)),
        )
