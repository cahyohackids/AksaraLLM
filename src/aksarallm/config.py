"""Model configuration helpers for AksaraLLM variants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class ModelConfig:
    """Typed transformer config consumed by ``aksarallm.model``."""

    vocab_size: int
    dim: int
    ffn_dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    max_seq_len: int
    norm_eps: float = 1e-6
    rope_theta: float = 10000.0


CONFIGS: Dict[str, Dict[str, Any]] = {
    "aksarallm-200m": {
        "vocab_size": 32000,
        "n_embd": 1024,
        "n_inner": 2816,
        "n_layers": 16,
        "n_heads": 8,
        "n_kv_heads": 4,
        "max_seq_len": 2048,
    },
    "aksarallm-500m": {
        "vocab_size": 32000,
        "n_embd": 1280,
        "n_inner": 3520,
        "n_layers": 20,
        "n_heads": 10,
        "n_kv_heads": 5,
        "max_seq_len": 2048,
    },
    "aksarallm-1.5b": {
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "sft_dataset": "AksaraLLM/aksara-mega-sft-v5",
        "dpo_dataset": "AksaraLLM/aksara-dpo-id-v4",
        "vocab_size": 151936,
        "n_embd": 1536,
        "n_inner": 8960,
        "n_layers": 28,
        "n_heads": 12,
        "n_kv_heads": 2,
        "max_seq_len": 32768,
        "rope_theta": 1000000.0,
    },
    "aksarallm-7b": {
        "vocab_size": 32000,
        "n_embd": 4096,
        "n_inner": 11008,
        "n_layers": 32,
        "n_heads": 32,
        "n_kv_heads": 8,
        "max_seq_len": 8192,
    },
    "aksarallm-20b": {
        "vocab_size": 131072,
        "n_embd": 6144,
        "n_inner": 16384,
        "n_layers": 42,
        "n_heads": 48,
        "n_kv_heads": 8,
        "max_seq_len": 8192,
        "rope_theta": 1000000.0,
    },
}


ALIASES = {
    "200m": "aksarallm-200m",
    "500m": "aksarallm-500m",
    "1b": "aksarallm-1.5b",
    "1.5b": "aksarallm-1.5b",
    "7b": "aksarallm-7b",
    "20b": "aksarallm-20b",
}


def _normalize_config(name: str, raw: Dict[str, Any]) -> ModelConfig:
    normalized = dict(raw)
    if "dim" not in normalized and "n_embd" in normalized:
        normalized["dim"] = normalized["n_embd"]
    if "ffn_dim" not in normalized and "n_inner" in normalized:
        normalized["ffn_dim"] = normalized["n_inner"]

    required = (
        "vocab_size",
        "dim",
        "ffn_dim",
        "n_layers",
        "n_heads",
        "n_kv_heads",
        "max_seq_len",
    )
    missing = [field for field in required if field not in normalized]
    if missing:
        raise ValueError(f"Config '{name}' is missing required fields: {', '.join(missing)}")

    return ModelConfig(
        vocab_size=int(normalized["vocab_size"]),
        dim=int(normalized["dim"]),
        ffn_dim=int(normalized["ffn_dim"]),
        n_layers=int(normalized["n_layers"]),
        n_heads=int(normalized["n_heads"]),
        n_kv_heads=int(normalized["n_kv_heads"]),
        max_seq_len=int(normalized["max_seq_len"]),
        norm_eps=float(normalized.get("norm_eps", 1e-6)),
        rope_theta=float(normalized.get("rope_theta", 10000.0)),
    )


def get_config(name: str) -> ModelConfig:
    """Return a typed config for a known AksaraLLM size alias or full name."""

    key = ALIASES.get(name, name)
    if key not in CONFIGS:
        known = ", ".join(sorted(set(CONFIGS) | set(ALIASES)))
        raise KeyError(f"Unknown config '{name}'. Known configs: {known}")
    return _normalize_config(key, CONFIGS[key])


__all__ = ["ALIASES", "CONFIGS", "ModelConfig", "get_config"]
