"""RWKV-7 "Goose" (x070) model configuration.

Reference: https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_demo.py
"""

import math

from transformers import PretrainedConfig

ATTN_CLASSES = {
    "global_value_normalization": "global_value_normalization",
}


def _lora_dim(coef: float, hidden_size: int, multiple_of: int = 32, minimum: int = 32) -> int:
    """Compute LoRA intermediate dim the same way as RWKV-LM x070 init."""
    return max(minimum, int(round(coef * (hidden_size ** 0.5) / multiple_of)) * multiple_of)


class Rwkv7Config(PretrainedConfig):
    """Configuration for the RWKV-7 "Goose" (x070) model.

    RWKV-7 is a 100% RNN with no kv-cache. The recurrent state has two parts per layer:
      - a vector state of shape (2, hidden_size): last x for att time-shift & ffn time-shift
      - an attention state matrix of shape (num_heads, head_size, head_size)
    """

    model_type = "rwkv7"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {
        "hidden_size": "hidden_size",
        "num_hidden_layers": "num_hidden_layers",
        "num_attention_heads": "num_attention_heads",
    }

    def __init__(
        self,
        vocab_size: int = 65536,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        head_size: int = 64,
        intermediate_size: int = None,
        # LoRA intermediate dims (default = computed from hidden_size, matching RWKV-LM x070)
        decay_lora_size: int = None,
        aaa_lora_size: int = None,
        mv_lora_size: int = None,
        gate_lora_size: int = None,
        # norm / activation
        layer_norm_eps: float = 1e-5,
        group_norm_eps: float = 64e-5,
        # training niceties
        use_cache: bool = True,
        # torchscript / gradient checkpointing
        torchscript: bool = False,
        use_grad_checkpoint: bool = False,
        # numerics
        wkv_dtype: str = "float32",
        **kwargs,
    ):
        assert hidden_size % head_size == 0, "hidden_size must be divisible by head_size"
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.head_size = head_size
        # attention: dim_att == hidden_size in x070; derived, kept for HF interop
        self.num_attention_heads = hidden_size // head_size
        # ffn
        self.intermediate_size = (
            intermediate_size if intermediate_size is not None else hidden_size * 4
        )
        # LoRA dims follow the RWKV-LM x070 formula; overridable for pretrained variants
        self.decay_lora_size = decay_lora_size or _lora_dim(2.5, hidden_size)
        self.aaa_lora_size = aaa_lora_size or _lora_dim(2.5, hidden_size)
        self.mv_lora_size = mv_lora_size or _lora_dim(1.7, hidden_size)
        self.gate_lora_size = gate_lora_size or _lora_dim(5.0, hidden_size)
        # norms
        self.layer_norm_eps = layer_norm_eps
        self.group_norm_eps = group_norm_eps
        # misc
        self.use_cache = use_cache
        self.use_grad_checkpoint = use_grad_checkpoint
        self.wkv_dtype = wkv_dtype

        # RWKV-7 is a non-transformer (linear/RNN) model; expose for downstream tooling
        self.is_decoder = False
        self.is_encoder_decoder = False

        super().__init__(
            torchscript=torchscript,
            tie_word_embeddings=kwargs.pop("tie_word_embeddings", False),
            **kwargs,
        )

    def to_dict(self):
        out = super().to_dict()
        # keep the derived field explicit in saved configs
        out["num_attention_heads"] = self.hidden_size // self.head_size
        return out
