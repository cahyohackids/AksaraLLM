#!/usr/bin/env python3
"""Pre-flight gate for an AksaraLLM 20B pretrain launch.

Runs a sequence of read-only checks and **fails loudly** if any gate is
not satisfied. Intended to be invoked as the last step before
``scripts/train_20b_pretrain.py`` on a TPU VM. Exit code 0 means "safe
to burn TPU-hours on this run"; any non-zero exit means do not launch.

The gates mirror the checklist at the end of
``AksaraLLM_20B_Readiness_Audit.md``:

1. JSON config parses and param count math checks out (≥ 20 B).
2. Python ``CONFIGS["aksarallm-20b"]`` loads from the same JSON
   (single source of truth).
3. Tokenizer exists at the requested HF path and has the right
   ``vocab_size`` + named special tokens at the pinned IDs.
4. Corpus glob resolves to at least one Parquet shard and a
   ``manifest.json`` exists in the parent directory.
5. Manifest claims ``>= --min-tokens`` realised tokens.
6. W&B is reachable (or explicitly skipped with ``--no-wandb``).
7. Output dir is writable (GCS or local), and Orbax can lay down a
   probe checkpoint at ``<output_dir>/_preflight_probe/`` (deleted on
   success).
8. The TPU topology has the expected ``jax.device_count()`` and the
   requested ``(dp_size, tp_size)`` evenly divides it.

Usage::

    python scripts/preflight_20b.py \
        --config configs/aksara_20b_dense.json \
        --tokenizer Ezekiel999/aksara-tokenizer-20b \
        --corpus-glob 'gs://aksarallm-corpus/pretrain/*/*.parquet' \
        --output-dir gs://aksarallm-checkpoints/20b \
        --expected-chips 128 --tp-size 4 \
        --min-tokens 400000000000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable


class Gate:
    """A single pre-flight gate."""

    def __init__(self, name: str):
        self.name = name
        self.ok = False
        self.detail = ""

    def passed(self, detail: str = "") -> None:
        self.ok = True
        self.detail = detail

    def failed(self, detail: str) -> None:
        self.ok = False
        self.detail = detail

    def __str__(self) -> str:
        sigil = "PASS" if self.ok else "FAIL"
        return f"  [{sigil}] {self.name}: {self.detail}"


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------


def _check_config(path: Path) -> Gate:
    g = Gate("config JSON parses + 20 B param math")
    try:
        with path.open() as f:
            cfg = json.load(f)
    except Exception as exc:
        g.failed(f"cannot read {path}: {exc}")
        return g
    try:
        arch = cfg["architecture"]
        V = arch["vocab_size"]
        d = arch["n_embd"]
        ffn = arch["n_inner"]
        L = arch["n_layers"]
        nh = arch["n_heads"]
        nkv = arch["n_kv_heads"]
        dh = arch["head_dim"]
        tied = bool(arch.get("tie_embeddings", True))
    except KeyError as exc:
        g.failed(f"missing key in architecture: {exc}")
        return g

    attn = 2 * d * (nh * dh) + 2 * d * (nkv * dh)  # Q,K,V,O
    mlp = 3 * d * ffn
    per_layer = attn + mlp + 2 * d
    total = L * per_layer + V * d + d
    if not tied:
        total += V * d

    claimed = cfg.get("estimated_params", 0)
    if total < 20_000_000_000:
        g.failed(
            f"computed params = {total:,} (< 20 B). Increase n_inner or n_layers."
        )
        return g
    if claimed and abs(claimed - total) > total * 0.01:
        g.failed(
            f"estimated_params={claimed:,} disagrees with computed={total:,} "
            "by >1 %. Fix the config."
        )
        return g
    g.passed(f"computed {total:,} params (~{total/1e9:.2f} B)")
    return g


def _check_python_config_matches_json(json_path: Path) -> Gate:
    g = Gate("src/aksarallm/config.py loads from the same JSON")
    try:
        sys.path.insert(0, str(json_path.parent.parent / "src"))
        from aksarallm.config import get_config  # type: ignore
    except Exception as exc:
        g.failed(f"cannot import aksarallm.config: {exc}")
        return g
    try:
        c = get_config("20b")
    except Exception as exc:
        g.failed(f"get_config('20b') raised: {exc}")
        return g
    with json_path.open() as f:
        arch = json.load(f)["architecture"]
    if (
        c.vocab_size != arch["vocab_size"]
        or c.dim != arch["n_embd"]
        or c.ffn_dim != arch["n_inner"]
        or c.n_layers != arch["n_layers"]
        or c.n_heads != arch["n_heads"]
        or c.n_kv_heads != arch["n_kv_heads"]
    ):
        g.failed(
            "Python CONFIGS['aksarallm-20b'] diverges from JSON. "
            "Re-run the import path in config.py."
        )
        return g
    g.passed("Python config matches JSON")
    return g


def _check_tokenizer(tokenizer_id: str) -> Gate:
    g = Gate(f"tokenizer '{tokenizer_id}' loads with 131072 vocab + pinned specials")
    try:
        from transformers import AutoTokenizer  # type: ignore
    except Exception as exc:
        g.failed(f"transformers not installed: {exc}")
        return g
    try:
        tok = AutoTokenizer.from_pretrained(tokenizer_id)
    except Exception as exc:
        g.failed(f"load failed: {exc}")
        return g
    expected_specials = [
        ("<|pad|>", 0), ("<|bos|>", 1), ("<|eos|>", 2), ("<|unk|>", 3),
        ("<|system|>", 4), ("<|user|>", 5), ("<|assistant|>", 6), ("<|tool|>", 7),
        ("<|im_start|>", 8), ("<|im_end|>", 9),
    ]
    mismatches = []
    for tok_str, expected_id in expected_specials:
        got = tok.convert_tokens_to_ids(tok_str)
        if got != expected_id:
            mismatches.append(f"{tok_str} expected {expected_id}, got {got}")
    if tok.vocab_size != 131072 and len(tok) != 131072:
        mismatches.append(f"vocab_size={tok.vocab_size} / len={len(tok)} != 131072")
    if mismatches:
        g.failed("; ".join(mismatches))
        return g
    g.passed("vocab=131072 and pinned special IDs OK")
    return g


def _list_glob(glob: str) -> list[str]:
    if glob.startswith("gs://"):
        try:
            import gcsfs  # type: ignore
            fs = gcsfs.GCSFileSystem()
            return fs.glob(glob)
        except Exception:
            return []
    from glob import glob as _glob
    return _glob(glob)


def _check_corpus(corpus_glob: str, min_tokens: int) -> Gate:
    g = Gate(f"corpus glob resolves + manifest.json reports >= {min_tokens:,} tokens")
    shards = _list_glob(corpus_glob)
    if not shards:
        g.failed(f"glob '{corpus_glob}' matched 0 files")
        return g

    # Find the manifest, which should live in the parent dir of the shards.
    # Both local paths and gs:// paths are handled.
    first = shards[0]
    if first.startswith("gs://") or first.startswith("gcs://"):
        try:
            import gcsfs  # type: ignore
            fs = gcsfs.GCSFileSystem()
            # Walk up until we find manifest.json
            base = first
            for _ in range(4):
                base = base.rsplit("/", 1)[0]
                cand = f"{base}/manifest.json"
                if fs.exists(cand):
                    with fs.open(cand, "r") as f:
                        manifest = json.load(f)
                    break
            else:
                g.failed(f"could not locate manifest.json near {first}")
                return g
        except Exception as exc:
            g.failed(f"gcsfs error: {exc}")
            return g
    else:
        p = Path(first)
        for _ in range(4):
            p = p.parent
            cand = p / "manifest.json"
            if cand.is_file():
                manifest = json.loads(cand.read_text())
                break
        else:
            g.failed(f"could not locate manifest.json near {first}")
            return g

    total_tokens = sum(
        int(s.get("kept_tokens", 0)) for s in manifest.get("per_source_stats", [])
    )
    if total_tokens < min_tokens:
        g.failed(
            f"manifest reports {total_tokens:,} kept tokens "
            f"(< required {min_tokens:,})"
        )
        return g
    g.passed(f"{len(shards)} shards, {total_tokens:,} kept tokens")
    return g


def _check_wandb(skip: bool) -> Gate:
    g = Gate("W&B reachable")
    if skip:
        g.passed("skipped by flag")
        return g
    try:
        import wandb  # type: ignore
        api = wandb.Api()
        _ = api.viewer  # triggers an auth check
        g.passed(f"logged in as {api.viewer['username']}")
    except Exception as exc:
        g.failed(f"wandb auth/check failed: {exc}")
    return g


def _check_output_dir(output_dir: str) -> Gate:
    g = Gate(f"output_dir writable: {output_dir}")
    try:
        if output_dir.startswith("gs://"):
            import gcsfs  # type: ignore
            fs = gcsfs.GCSFileSystem()
            probe = f"{output_dir.rstrip('/')}/_preflight_probe/marker"
            with fs.open(probe, "w") as f:
                f.write("ok")
            fs.rm(probe)
        else:
            p = Path(output_dir) / "_preflight_probe"
            p.mkdir(parents=True, exist_ok=True)
            (p / "marker").write_text("ok")
            (p / "marker").unlink()
            p.rmdir()
        g.passed("write probe OK")
    except Exception as exc:
        g.failed(f"write probe failed: {exc}")
    return g


def _check_topology(expected_chips: int, tp_size: int, dp_size: int | None) -> Gate:
    g = Gate(f"TPU topology has {expected_chips} chips, divisible by tp={tp_size}")
    try:
        import jax  # type: ignore
    except Exception as exc:
        g.failed(f"JAX unavailable: {exc}")
        return g
    try:
        actual = jax.device_count()
    except Exception as exc:
        g.failed(f"jax.device_count() raised: {exc}")
        return g
    if actual != expected_chips:
        g.failed(f"expected {expected_chips} chips, got {actual}")
        return g
    if actual % tp_size != 0:
        g.failed(f"{actual} chips not divisible by tp_size={tp_size}")
        return g
    if dp_size is not None and dp_size * tp_size != actual:
        g.failed(f"dp_size*tp_size ({dp_size}*{tp_size}) != {actual}")
        return g
    g.passed(f"{actual} chips, tp={tp_size}")
    return g


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--corpus-glob", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-tokens", type=int, default=400_000_000_000)
    parser.add_argument("--expected-chips", type=int, default=None,
                        help="If provided, assert jax.device_count() matches.")
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--dp-size", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()

    gates: Iterable[Gate] = [
        _check_config(config_path),
        _check_python_config_matches_json(config_path),
        _check_tokenizer(args.tokenizer),
        _check_corpus(args.corpus_glob, args.min_tokens),
        _check_wandb(skip=args.no_wandb),
        _check_output_dir(args.output_dir),
    ]
    if args.expected_chips is not None:
        gates = list(gates) + [
            _check_topology(args.expected_chips, args.tp_size, args.dp_size)
        ]

    print("AksaraLLM 20B pre-flight:")
    all_ok = True
    for g in gates:
        print(g)
        all_ok = all_ok and g.ok

    if all_ok:
        print("\nALL GATES PASSED — safe to launch.")
        return 0
    print("\nONE OR MORE GATES FAILED — DO NOT LAUNCH.")
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
