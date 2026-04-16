"""AksaraLLM core package exports."""

from .config import ALIASES, CONFIGS, ModelConfig, get_config

__version__ = "2.0.0"
__author__ = "AksaraLLM Team"

__all__ = [
    "ALIASES",
    "AksaraLLM",
    "AksaraTokenizer",
    "CONFIGS",
    "ModelConfig",
    "get_config",
]


def __getattr__(name: str):
    if name == "AksaraLLM":
        from .model import AksaraLLM as _AksaraLLM

        return _AksaraLLM
    if name == "AksaraTokenizer":
        from .tokenizer_utils import AksaraTokenizer as _AksaraTokenizer

        return _AksaraTokenizer
    raise AttributeError(f"module 'aksarallm' has no attribute {name!r}")
