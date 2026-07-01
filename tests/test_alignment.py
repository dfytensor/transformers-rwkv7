"""Numerical alignment test against the OFFICIAL ``rwkv_v7_demo.py`` pure-PyTorch reference.

We rebuild the reference model directly from the official demo source (the slow-PyTorch
WKV path, ``USE_CUDA_KERNEL=False``) as an *independent* implementation, load the same
pretrained checkpoint into both, and compare logits on identical input.

Pass criterion: max relative deviation < 1e-3 (fp32, different op-order in the recurrence).
"""

import os
import sys
import math
import types

import torch
import torch.nn as nn
from torch.nn import functional as F

sys.path.insert(0, os.path.dirname(__file__))
from transformers_rwkv7 import Rwkv7Config, Rwkv7ForCausalLM  # noqa: E402

MODEL_PATH = os.environ.get(
    "RWKV7_TEST_MODEL",
    r"F:\rwkv\models\rwkv7-g1d-0.1b-20260129-ctx8192.pth",
)

# ======================================================================================================================
# OFFICIAL REFERENCE — verbatim from BlinkDL/RWKV-LM RWKV-v7/rwkv_v7_demo.py (USE_CUDA_KERNEL=False path)
# ======================================================================================================================

HEAD_SIZE = 64


def ref_RWKV7_OP(r, w, k, v, a, b):  # noqa: N802 — keep official name
    B, T, C = r.size()
    H = C // HEAD_SIZE
    N = HEAD_SIZE
    r = r.view(B, T, H, N).float()
    k = k.view(B, T, H, N).float()
    v = v.view(B, T, H, N).float()
    a = a.view(B, T, H, N).float()
    b = b.view(B, T, H, N).float()
    w = torch.exp(-torch.exp(w.view(B, T, H, N).float()))
    out = torch.zeros((B, T, H, N), device=r.device, dtype=torch.float)
    state = torch.zeros((B, H, N, N), device=r.device, dtype=torch.float)
    for t in range(T):
        kk = k[:, t, :].view(B, H, 1, N)
        rr = r[:, t, :].view(B, H, N, 1)
        vv = v[:, t, :].view(B, H, N, 1)
        aa = a[:, t, :].view(B, H, N, 1)
        bb = b[:, t, :].view(B, H, 1, N)
        state = state * w[:, t, :, None, :] + state @ aa @ bb + vv @ kk
        out[:, t, :] = (state @ rr).view(B, H, N)
    return out.view(B, T, C)


class RefRWKV_Tmix_x070(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        C = args.n_embd
        H, N = self.n_head, self.head_size
        D_DECAY_LORA, D_AAA_LORA, D_MV_LORA, D_GATE_LORA = (
            args.D_DECAY_LORA, args.D_AAA_LORA, args.D_MV_LORA, args.D_GATE_LORA)
        self.x_r = nn.Parameter(torch.empty(1, 1, C))
        self.x_w = nn.Parameter(torch.empty(1, 1, C))
        self.x_k = nn.Parameter(torch.empty(1, 1, C))
        self.x_v = nn.Parameter(torch.empty(1, 1, C))
        self.x_a = nn.Parameter(torch.empty(1, 1, C))
        self.x_g = nn.Parameter(torch.empty(1, 1, C))
        self.w0 = nn.Parameter(torch.empty(1, 1, C))
        self.w1 = nn.Parameter(torch.empty(C, D_DECAY_LORA))
        self.w2 = nn.Parameter(torch.empty(D_DECAY_LORA, C))
        self.a0 = nn.Parameter(torch.empty(1, 1, C))
        self.a1 = nn.Parameter(torch.empty(C, D_AAA_LORA))
        self.a2 = nn.Parameter(torch.empty(D_AAA_LORA, C))
        self.v0 = nn.Parameter(torch.empty(1, 1, C))
        self.v1 = nn.Parameter(torch.empty(C, D_MV_LORA))
        self.v2 = nn.Parameter(torch.empty(D_MV_LORA, C))
        self.g1 = nn.Parameter(torch.empty(C, D_GATE_LORA))
        self.g2 = nn.Parameter(torch.empty(D_GATE_LORA, C))
        self.k_k = nn.Parameter(torch.empty(1, 1, C))
        self.k_a = nn.Parameter(torch.empty(1, 1, C))
        self.r_k = nn.Parameter(torch.empty(H, N))
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        self.ln_x = nn.GroupNorm(H, C, eps=64e-5)

    def forward(self, x, v_first):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x
        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g
        r = self.receptance(xr)
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
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)
        x = ref_RWKV7_OP(r, w, k, v, -kk, kk * a)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)
        x = x + ((r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k).sum(dim=-1, keepdim=True) * v.view(B, T, H, -1)).view(B, T, C)
        x = self.output(x * g)
        return x, v_first


class RefRWKV_CMix_x070(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        C = args.n_embd
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.x_k = nn.Parameter(torch.empty(1, 1, C))
        self.key = nn.Linear(C, args.dim_ffn, bias=False)
        self.value = nn.Linear(args.dim_ffn, C, bias=False)

    def forward(self, x):
        xx = self.time_shift(x) - x
        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)


class RefBlock(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.layer_id = layer_id
        self.ln0 = nn.LayerNorm(args.n_embd)
        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)
        self.att = RefRWKV_Tmix_x070(args, layer_id)
        self.ffn = RefRWKV_CMix_x070(args, layer_id)

    def forward(self, x, v_first):
        if self.layer_id == 0:
            x = self.ln0(x)
        xx, v_first = self.att(self.ln1(x), v_first)
        x = x + xx
        x = x + self.ffn(self.ln2(x))
        return x, v_first


class RefRWKV(nn.Module):
    def __init__(self, args):
        super().__init__()
        args.dim_att = args.n_embd
        args.dim_ffn = args.n_embd * 4
        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        self.blocks = nn.ModuleList([RefBlock(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

    def forward(self, idx):
        x = self.emb(idx)
        v_first = torch.empty_like(x)
        for block in self.blocks:
            x, v_first = block(x, v_first)
        x = self.ln_out(x)
        return self.head(x)


# ======================================================================================================================
# Test driver
# ======================================================================================================================

def main():
    if not os.path.isfile(MODEL_PATH):
        print(f"SKIP: model not found at {MODEL_PATH} (set RWKV7_TEST_MODEL env)")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    print(f"device = {device}")

    # ---- config inferred from checkpoint ----
    n_layer = 12
    n_embd = 768
    config = Rwkv7Config(
        vocab_size=65536,
        hidden_size=n_embd,
        num_hidden_layers=n_layer,
        head_size=64,
        intermediate_size=3072,
        decay_lora_size=64,
        aaa_lora_size=64,
        mv_lora_size=32,
        gate_lora_size=128,
    )

    raw = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)

    # ---- reference model ----
    args = types.SimpleNamespace(
        n_layer=n_layer, n_embd=n_embd, head_size_a=64, vocab_size=65536,
        D_DECAY_LORA=64, D_AAA_LORA=64, D_MV_LORA=32, D_GATE_LORA=128,
    )
    ref = RefRWKV(args).float().eval()
    missing, unexpected = ref.load_state_dict(raw, strict=False)
    ref = ref.to(device)

    # ---- our model: remap checkpoint keys (add `rwkv7.` prefix to body modules) ----
    remapped = {}
    for k, v in raw.items():
        if k.startswith("head."):
            remapped[k] = v
        elif k.startswith("emb.") or k.startswith("blocks.") or k.startswith("ln_out"):
            remapped["rwkv7." + k] = v
        else:
            remapped[k] = v
    ours = Rwkv7ForCausalLM(config).float().eval()
    missing2, unexpected2 = ours.load_state_dict(remapped, strict=False)
    ours = ours.to(device)

    print(f"ref missing/unexpected: {len(missing)}/{len(unexpected)}")
    print(f"ours missing/unexpected: {len(missing2)}/{len(unexpected2)}")
    # ours should only be missing ln0 in blocks>0 (which the ckpt has but we don't use) — actually those are *unexpected* for us
    real_missing = [m for m in missing2 if "ln0" not in m]
    assert not real_missing, f"unexpected missing keys in our model: {real_missing}"

    # ---- compare logits ----
    input_ids = torch.tensor([[0, 5248, 478, 2214, 30, 15025, 19, 358, 4457, 311]], device=device)
    with torch.no_grad():
        ref_logits = ref(input_ids).float()
        our_logits = ours(input_ids).logits.float()

    max_abs = (our_logits - ref_logits).abs().max().item()
    ref_scale = ref_logits.abs().max().item()
    rel = max_abs / ref_scale
    # argmax agreement
    agree = (our_logits.argmax(-1) == ref_logits.argmax(-1)).float().mean().item()
    print(f"\ninput:        {input_ids.shape}")
    print(f"ref last-token top-5: {torch.topk(ref_logits[0,-1], 5).indices.tolist()}")
    print(f"our last-token top-5: {torch.topk(our_logits[0,-1], 5).indices.tolist()}")
    print(f"max|Δ|:       {max_abs:.6e}")
    print(f"ref scale:    {ref_scale:.4f}")
    print(f"rel dev:      {rel:.6e}")
    print(f"argmax agree: {agree*100:.2f}%")

    assert rel < 1e-3, f"logits diverge: rel dev {rel:.3e} >= 1e-3"
    assert agree > 0.98, f"argmax agreement too low: {agree:.3f}"
    print("\nALIGNMENT TEST PASSED")


if __name__ == "__main__":
    main()
