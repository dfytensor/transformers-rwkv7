from .configuration_rwkv7 import Rwkv7Config
from .modeling_rwkv7 import (
    Rwkv7PreTrainedModel,
    Rwkv7Model,
    Rwkv7ForCausalLM,
    Rwkv7State,
    Rwkv7Block,
    Rwkv7TimeMix,
    Rwkv7ChannelMix,
)
from .tokenization_rwkv7 import Rwkv7Tokenizer


def _register_auto_classes():
    """Register Rwkv7 classes with transformers' Auto system (idempotent).

    After ``import transformers_rwkv7`` you can use:
        AutoConfig.from_pretrained(...) / AutoModelForCausalLM.from_pretrained(...)
    without passing ``trust_remote_code`` (local use).
    """
    try:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        AutoConfig.register("rwkv7", Rwkv7Config)
        # transformers>=5.0 changed the signature to (config_class, model_class)
        try:
            AutoModelForCausalLM.register(Rwkv7Config, Rwkv7ForCausalLM)
        except TypeError:
            AutoModelForCausalLM.register("rwkv7", Rwkv7ForCausalLM)
        try:
            AutoTokenizer.register("rwkv7", fast_tokenizer_class=None, tokenizer_cls=Rwkv7Tokenizer)
        except Exception:
            pass
    except Exception:
        # transformers not installed at import time — registration is best-effort
        pass


_register_auto_classes()

__version__ = "0.1.0"

__all__ = [
    "Rwkv7Config",
    "Rwkv7PreTrainedModel",
    "Rwkv7Model",
    "Rwkv7ForCausalLM",
    "Rwkv7State",
    "Rwkv7Block",
    "Rwkv7TimeMix",
    "Rwkv7ChannelMix",
    "Rwkv7Tokenizer",
    "__version__",
]
