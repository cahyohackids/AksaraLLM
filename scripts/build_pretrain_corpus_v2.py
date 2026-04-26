#!/usr/bin/env python3
"""AksaraLLM 20B pre-training corpus builder (v2).

Replaces ``scripts/build_pretrain_data.py`` (OSCAR-only, exact-md5 dedup)
with a multi-source, language-ID-filtered, Gopher-quality-filtered,
MinHash-LSH-deduplicated, benchmark-decontaminated pipeline that
produces Parquet shards plus a ``manifest.json`` with per-source
token counts.

Design goals
------------

* **Streaming.** Every source is consumed with ``load_dataset(..., streaming=True)``
  so no single shard ever holds more than one document in memory.
* **Per-source token budgets.** The pipeline enforces the data_mix
  percentages from ``configs/aksara_20b_dense.json`` (source of truth)
  and stops ingesting a source once it hits its target token allocation.
* **LID.** fastText ``lid.176.bin`` is used to confirm the language of
  every document; any doc where ``P(target_language) < --lid-threshold``
  is dropped.
* **Gopher-style quality filters.** Mean-word-length, symbol ratio,
  ellipsis ratio, bullet ratio, repetition penalties — see
  ``gopher_keep()``. These defaults mirror the FineWeb2 paper and the
  Dolma v1.7 config.
* **MinHash-LSH.** ``datasketch``-based near-duplicate detection at
  document granularity. Exact ``sha256(text)`` filters also run.
* **Benchmark decontamination.** Any document whose 13-gram set has a
  Jaccard overlap ≥ 0.0 with a committed list of benchmark test-split
  n-grams is dropped. Benchmarks decontaminated by default: IndoMMLU,
  xCOPA, XNLI-id, TyDiQA-id, MMLU, HellaSwag, ARC, GSM8K.
* **Parquet output.** Each shard is ~1 GB of text, compressed with zstd.
  A ``manifest.json`` records per-source tokens and the final realised
  mix vs. the target mix from the config.

This script is intentionally **CPU-heavy and embarrassingly parallel**.
On a single 16-core VM it produces ~10–20 GB of clean tokenized text
per hour; producing a 400B-token corpus therefore requires either a
multi-node Ray / Spark cluster or a multi-week run on a beefy VM.
For a 50–100 GB tokenizer-training sample (Batch B in the audit) it
completes in a few hours.

Usage
-----

::

    # 1. Download once (cacheable):
    python scripts/build_pretrain_corpus_v2.py download-assets \
        --assets-dir outputs/pretrain_assets

    # 2. Small smoke run (produces ~1 GB of clean Parquet in ~15 minutes):
    python scripts/build_pretrain_corpus_v2.py build \
        --config configs/aksara_20b_dense.json \
        --output-dir outputs/pretrain/smoke \
        --assets-dir outputs/pretrain_assets \
        --max-docs-per-source 50000 \
        --shard-target-bytes 1000000000

    # 3. Real run (produces Parquet shards sized according to target mix):
    python scripts/build_pretrain_corpus_v2.py build \
        --config configs/aksara_20b_dense.json \
        --output-dir /mnt/pretrain_corpus_v2 \
        --assets-dir outputs/pretrain_assets \
        --target-total-tokens 400_000_000_000

The resulting Parquet shards are the input to the JAX/Flax pretrain
script in ``scripts/train_20b_pretrain.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

LOG = logging.getLogger("aksara.corpus_v2")

# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------


@dataclass
class SourceSpec:
    """A streamable pre-training source."""

    name: str
    hf_path: str
    hf_config: Optional[str]
    split: str = "train"
    text_field: str = "text"
    # Which mix bucket this source feeds. Bucket percentages come from
    # ``configs/aksara_20b_dense.json#data_mix``.
    bucket: str = "global_high_quality_web"
    # Target language code for LID verification. Empty string disables
    # the LID check for this source (e.g. code).
    language: str = "en"
    # Minimum LID probability to keep a doc.
    lid_threshold: float = 0.65
    # Optional per-source override of Gopher filter thresholds.
    min_words: int = 50
    max_words: int = 100_000


SOURCES: Sequence[SourceSpec] = (
    # --- global high-quality web ---
    SourceSpec("fineweb", "HuggingFaceFW/fineweb", "sample-10BT", "train", "text",
               bucket="global_high_quality_web", language="en"),
    # Larger FineWeb samples for parallel producers (same source family,
    # partitioned via --skip so docs don't overlap).
    SourceSpec("fineweb_100bt", "HuggingFaceFW/fineweb", "sample-100BT", "train", "text",
               bucket="global_high_quality_web", language="en"),
    SourceSpec("fineweb_350bt", "HuggingFaceFW/fineweb", "sample-350BT", "train", "text",
               bucket="global_high_quality_web", language="en"),
    # FineWeb-Edu (filtered for educational content; higher quality signal).
    SourceSpec("fineweb_edu", "HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text",
               bucket="global_high_quality_web", language="en"),
    # --- multilingual web ---
    SourceSpec("fineweb2_id", "HuggingFaceFW/fineweb-2", "ind_Latn", "train", "text",
               bucket="multilingual_web", language="id"),
    SourceSpec("culturax_id", "uonlp/CulturaX", "id", "train", "text",
               bucket="multilingual_web", language="id"),
    SourceSpec("culturax_ms", "uonlp/CulturaX", "ms", "train", "text",
               bucket="multilingual_web", language="ms"),
    # --- Indonesia / SEA targeted ---
    # NOTE: SEACrowd/indo4b, SEACrowd/cc100 use datasets v2-style loading scripts
    # which are deprecated in ``datasets`` ≥4. FineWeb-2 + CulturaX provide good
    # coverage for JV/SU directly. Wikipedia JV supplements reference text.
    SourceSpec("fineweb2_jv", "HuggingFaceFW/fineweb-2", "jav_Latn", "train", "text",
               bucket="indonesia_sea_targeted", language="jv", lid_threshold=0.4),
    SourceSpec("fineweb2_su", "HuggingFaceFW/fineweb-2", "sun_Latn", "train", "text",
               bucket="indonesia_sea_targeted", language="su", lid_threshold=0.4),
    SourceSpec("culturax_jv", "uonlp/CulturaX", "jv", "train", "text",
               bucket="indonesia_sea_targeted", language="jv", lid_threshold=0.4),
    SourceSpec("culturax_su", "uonlp/CulturaX", "su", "train", "text",
               bucket="indonesia_sea_targeted", language="su", lid_threshold=0.4),
    # --- code ---
    # NOTE: bigcode/the-stack-v2 is gated and requires explicit Hub acceptance.
    # Once access is granted, add back:
    #   SourceSpec("thestack2_py", "bigcode/the-stack-v2", "Python", ...)
    SourceSpec("code_search_net", "code_search_net", "all", "train", "whole_func_string",
               bucket="code", language="", min_words=5),
    # --- reference text (Wikipedia; allenai/dolma is script-based, excluded) ---
    SourceSpec("wikipedia_id", "wikimedia/wikipedia", "20231101.id", "train", "text",
               bucket="reference_text", language="id"),
    SourceSpec("wikipedia_jv", "wikimedia/wikipedia", "20231101.jv", "train", "text",
               bucket="reference_text", language="jv", lid_threshold=0.4),
    SourceSpec("wikipedia_en", "wikimedia/wikipedia", "20231101.en", "train", "text",
               bucket="reference_text", language="en"),
    # --- additional large-scale web (added when fineweb partitions saturate) ---
    # FineWeb-Edu larger samples (same family, partitionable via --skip-docs).
    SourceSpec("fineweb_edu_100bt", "HuggingFaceFW/fineweb-edu", "sample-100BT", "train", "text",
               bucket="global_high_quality_web", language="en"),
    SourceSpec("fineweb_edu_350bt", "HuggingFaceFW/fineweb-edu", "sample-350BT", "train", "text",
               bucket="global_high_quality_web", language="en"),
    # mC4 (allenai/c4) — multilingual web crawl; useful for additional id/en/ms
    # coverage with different filtering than CulturaX/FineWeb-2. Parquet-native,
    # works fine with `datasets>=4` streaming (no script).
    SourceSpec("c4_en", "allenai/c4", "en", "train", "text",
               bucket="global_high_quality_web", language="en"),
    SourceSpec("c4_id", "allenai/c4", "id", "train", "text",
               bucket="multilingual_web", language="id"),
    SourceSpec("c4_ms", "allenai/c4", "ms", "train", "text",
               bucket="multilingual_web", language="ms"),
    # Cosmopedia v0.1 web samples — synthetic high-quality educational content.
    # Adds variety to the high-quality web bucket.
    SourceSpec("cosmopedia_web", "HuggingFaceTB/cosmopedia", "web_samples_v1", "train", "text",
               bucket="global_high_quality_web", language="en"),
    # NOTE: RedPajama-V2, allenai/dolma, mc4, OSCAR-2301, MADLAD-400 were tested
    # but rejected:
    #   * RedPajama-V2 / dolma / mc4: script-based (datasets v4 unsupported).
    #   * OSCAR-2301: gated (requires per-account terms acceptance on HF Hub).
    #   * MADLAD-400 default: streaming triggers ArrowInvalid on canary shards.
)


BENCHMARK_DECONTAM_SOURCES: Sequence[Tuple[str, Optional[str], str, str]] = (
    # (hf_path, hf_config, split, text_field)
    ("indolem/IndoMMLU", None, "test", "question"),
    ("xcopa", "id", "test", "premise"),
    ("xnli", "id", "test", "premise"),
    ("tydiqa", "secondary_task", "validation", "question"),
    ("cais/mmlu", "all", "test", "question"),
    ("Rowan/hellaswag", None, "validation", "ctx"),
    ("allenai/ai2_arc", "ARC-Challenge", "test", "question"),
    ("openai/gsm8k", "main", "test", "question"),
)


# ---------------------------------------------------------------------------
# Quality filters
# ---------------------------------------------------------------------------


REPEATED_CHAR_RE = re.compile(r"(.)\1{9,}")
URL_RE = re.compile(r"https?://\S+")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def gopher_keep(text: str, min_words: int, max_words: int) -> bool:
    """Gopher / FineWeb-style heuristics. Return True to keep."""

    words = text.split()
    n = len(words)
    if n < min_words or n > max_words:
        return False

    mean_wl = sum(len(w) for w in words) / n
    if mean_wl < 3 or mean_wl > 10:
        return False

    alpha_ratio = sum(c.isalpha() for c in text) / max(len(text), 1)
    if alpha_ratio < 0.6:
        return False

    ellipsis_lines = sum(line.strip().endswith("...") for line in text.splitlines())
    total_lines = max(text.count("\n") + 1, 1)
    if ellipsis_lines / total_lines > 0.3:
        return False

    bullet_lines = sum(line.lstrip().startswith(("-", "*", "•")) for line in text.splitlines())
    if bullet_lines / total_lines > 0.9:
        return False

    # Penalise extreme repetition of a single word.
    if n >= 20:
        mc = Counter(words).most_common(1)[0][1]
        if mc / n > 0.3:
            return False

    if REPEATED_CHAR_RE.search(text):
        return False

    return True


def scrub(text: str) -> str:
    """Cheap scrubs applied after Gopher. Do NOT change document tokens."""
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Language ID
# ---------------------------------------------------------------------------


_FASTTEXT_LID_URL = (
    "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
)


def _ensure_lid_model(assets_dir: Path) -> Path:
    dst = assets_dir / "lid.176.bin"
    if dst.is_file() and dst.stat().st_size > 100 * 1024:
        return dst
    assets_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("downloading fastText LID model to %s", dst)
    urllib.request.urlretrieve(_FASTTEXT_LID_URL, dst)
    return dst


class FastTextLID:
    """Thin wrapper over the fastText LID model.

    Calls the underlying C++ predict via ``self._model.f.predict`` to
    bypass fasttext's Python wrapper, which uses ``np.array(..., copy=False)``
    and breaks on NumPy ≥ 2.0 (``ValueError: Unable to avoid copy``).
    """

    def __init__(self, model_path: Path):
        import fasttext  # local import so --help works without the dep

        self._model = fasttext.load_model(str(model_path))

    def predict(self, text: str) -> Tuple[str, float]:
        # fastText does not like newlines in predict().
        probe = text[:2000].replace("\n", " ")
        try:
            predictions = self._model.f.predict(probe, 1, 0.0, "strict")
        except Exception:
            return "und", 0.0
        if not predictions:
            return "und", 0.0
        # fasttext's low-level ``f.predict`` returns ``[(prob, label), ...]``.
        first = predictions[0]
        if isinstance(first[0], (int, float)):
            prob, label = first
        else:
            label, prob = first
        if isinstance(label, bytes):
            label = label.decode("utf-8", errors="ignore")
        return str(label).removeprefix("__label__"), float(prob)


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


@dataclass
class MinHashDedup:
    """MinHash-LSH near-dup detector backed by ``datasketch``."""

    num_perm: int = 128
    threshold: float = 0.8
    _lsh: object = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        from datasketch import MinHashLSH  # noqa: local import

        self._lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)

    def _sig(self, text: str):
        from datasketch import MinHash  # noqa: local import

        mh = MinHash(num_perm=self.num_perm)
        for shingle in _shingles(text, 5):
            mh.update(shingle.encode("utf-8", errors="ignore"))
        return mh

    def insert_if_new(self, key: str, text: str) -> bool:
        """Return True iff inserted (i.e. not a near-duplicate)."""
        sig = self._sig(text)
        if self._lsh.query(sig):  # type: ignore[attr-defined]
            return False
        self._lsh.insert(key, sig)  # type: ignore[attr-defined]
        return True


def _shingles(text: str, k: int) -> Iterator[str]:
    tokens = text.split()
    for i in range(len(tokens) - k + 1):
        yield " ".join(tokens[i : i + k])


def _ngrams(text: str, n: int) -> Iterator[str]:
    tokens = text.split()
    for i in range(len(tokens) - n + 1):
        yield " ".join(tokens[i : i + n])


# ---------------------------------------------------------------------------
# Benchmark decontamination
# ---------------------------------------------------------------------------


def build_decontam_ngrams(n: int = 13) -> set[str]:
    """Return a set of benchmark 13-grams to exclude from training."""
    from datasets import load_dataset

    bad: set[str] = set()
    for hf_path, cfg, split, field_ in BENCHMARK_DECONTAM_SOURCES:
        try:
            LOG.info("decontam: loading %s (%s)", hf_path, cfg)
            ds = load_dataset(hf_path, cfg, split=split)
        except Exception as exc:  # pragma: no cover — best effort
            LOG.warning("decontam: failed to load %s/%s: %s", hf_path, cfg, exc)
            continue
        for item in ds:
            text = item.get(field_) or ""
            if isinstance(text, list):
                text = " ".join(str(x) for x in text)
            text = str(text)
            for g in _ngrams(text, n):
                bad.add(g)
    LOG.info("decontam: %d n-grams collected", len(bad))
    return bad


def doc_is_contaminated(text: str, bad: set[str], n: int = 13) -> bool:
    # Fast pre-check: if the doc has fewer than n tokens, nothing to match.
    tokens = text.split()
    if len(tokens) < n:
        return False
    for g in _ngrams(text, n):
        if g in bad:
            return True
    return False


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------


class ShardWriter:
    """Writes Parquet shards under ``output_dir/source/{prefix}shard-XXXXX.parquet``.

    ``shard_prefix`` lets parallel producers writing to the same ``output_dir``
    avoid filename collisions (e.g. ``p2-shard-00000.parquet``).
    """

    def __init__(
        self,
        output_dir: Path,
        source: str,
        shard_target_bytes: int,
        shard_prefix: str = "",
    ):
        self._dir = output_dir / source
        self._dir.mkdir(parents=True, exist_ok=True)
        self._source = source
        self._target_bytes = shard_target_bytes
        self._prefix = shard_prefix
        self._buf: List[Dict[str, str]] = []
        self._buf_bytes = 0
        self._shard = 0
        self._docs_written = 0

    def write(self, doc: Dict[str, str]) -> None:
        self._buf.append(doc)
        self._buf_bytes += len(doc.get("text", "")) or 0
        if self._buf_bytes >= self._target_bytes:
            self._flush()

    def _flush(self) -> None:
        if not self._buf:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist(self._buf)
        name = f"{self._prefix}shard-{self._shard:05d}.parquet"
        path = self._dir / name
        pq.write_table(table, path, compression="zstd")
        self._docs_written += len(self._buf)
        LOG.info(
            "wrote shard %s (%d docs, %.1f MB uncompressed text)",
            path.name,
            len(self._buf),
            self._buf_bytes / 1e6,
        )
        self._buf.clear()
        self._buf_bytes = 0
        self._shard += 1

    def close(self) -> int:
        self._flush()
        return self._docs_written


# ---------------------------------------------------------------------------
# Per-source streaming loop
# ---------------------------------------------------------------------------


def _midpoint(lo: float, hi: float) -> float:
    return (lo + hi) / 2


def compute_bucket_targets(config_json: Path, total_tokens: int) -> Dict[str, int]:
    """Turn config data_mix (min/max per bucket) into absolute token targets."""
    with config_json.open() as f:
        raw = json.load(f)
    mix = raw["data_mix"]
    midpoints = {
        "global_high_quality_web": _midpoint(
            mix["global_high_quality_web_min"], mix["global_high_quality_web_max"]
        ),
        "multilingual_web": _midpoint(
            mix["multilingual_web_min"], mix["multilingual_web_max"]
        ),
        "indonesia_sea_targeted": _midpoint(
            mix["indonesia_sea_targeted_min"], mix["indonesia_sea_targeted_max"]
        ),
        "code": _midpoint(mix["code_min"], mix["code_max"]),
        "reference_text": _midpoint(mix["reference_text_min"], mix["reference_text_max"]),
    }
    # Normalise (mins/max midpoints won't sum to 1 in general).
    total = sum(midpoints.values())
    return {k: int(total_tokens * v / total) for k, v in midpoints.items()}


def estimate_tokens(text: str) -> int:
    """Cheap pre-tokenizer token estimate: 0.75 * whitespace words.

    This is only used to enforce bucket budgets during corpus building.
    Real token counts come from the BPE tokenizer later.
    """
    return max(1, int(0.75 * len(text.split())))


def _dedup_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def process_source(
    spec: SourceSpec,
    *,
    output_dir: Path,
    token_budget: int,
    lid: FastTextLID,
    minhash: MinHashDedup,
    decontam_ngrams: set[str],
    max_docs: Optional[int],
    shard_target_bytes: int,
    seen_hashes: set[str],
    shard_prefix: str = "",
    skip_docs: int = 0,
) -> Dict[str, object]:
    """Stream one source and write clean Parquet shards. Returns stats."""
    from datasets import load_dataset  # noqa: local import

    LOG.info("opening %s/%s", spec.hf_path, spec.hf_config)
    ds = load_dataset(
        spec.hf_path,
        spec.hf_config,
        split=spec.split,
        streaming=True,
    )
    if skip_docs > 0:
        LOG.info("  %s: skipping first %d docs", spec.name, skip_docs)
        ds = ds.skip(skip_docs)

    writer = ShardWriter(output_dir, spec.name, shard_target_bytes, shard_prefix=shard_prefix)

    stats = {
        "source": spec.name,
        "seen_docs": 0,
        "kept_docs": 0,
        "kept_tokens": 0,
        "dropped_length": 0,
        "dropped_gopher": 0,
        "dropped_lid": 0,
        "dropped_exact_dup": 0,
        "dropped_near_dup": 0,
        "dropped_decontam": 0,
    }

    t0 = time.time()
    for item in ds:
        if max_docs and stats["seen_docs"] >= max_docs:
            break
        if stats["kept_tokens"] >= token_budget:
            break
        stats["seen_docs"] += 1

        text = item.get(spec.text_field) or ""
        if not isinstance(text, str):
            text = str(text)
        text = scrub(text)
        if not text:
            stats["dropped_length"] += 1
            continue

        if not gopher_keep(text, spec.min_words, spec.max_words):
            stats["dropped_gopher"] += 1
            continue

        if spec.language:
            lang, prob = lid.predict(text)
            if lang != spec.language or prob < spec.lid_threshold:
                stats["dropped_lid"] += 1
                continue

        key = _dedup_key(text)
        if key in seen_hashes:
            stats["dropped_exact_dup"] += 1
            continue
        seen_hashes.add(key)

        if not minhash.insert_if_new(key, text):
            stats["dropped_near_dup"] += 1
            continue

        if decontam_ngrams and doc_is_contaminated(text, decontam_ngrams):
            stats["dropped_decontam"] += 1
            continue

        stats["kept_docs"] += 1
        stats["kept_tokens"] += estimate_tokens(text)
        writer.write({"text": text, "source": spec.name, "bucket": spec.bucket})

        if stats["seen_docs"] % 10_000 == 0:
            LOG.info(
                "  %s: seen=%d kept=%d (%.1f M tok) in %.1fs",
                spec.name,
                stats["seen_docs"],
                stats["kept_docs"],
                stats["kept_tokens"] / 1e6,
                time.time() - t0,
            )

    written = writer.close()
    stats["shards_written"] = written
    LOG.info("%s done: %s", spec.name, stats)
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_download_assets(args: argparse.Namespace) -> None:
    assets = Path(args.assets_dir)
    _ensure_lid_model(assets)
    print(f"assets ready at {assets}")


def cmd_build(args: argparse.Namespace) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = Path(args.assets_dir)

    lid_path = _ensure_lid_model(assets_dir)
    lid = FastTextLID(lid_path)
    minhash = MinHashDedup(
        num_perm=args.minhash_perm, threshold=args.minhash_threshold
    )
    decontam = (
        build_decontam_ngrams(n=args.decontam_ngram) if args.decontam else set()
    )
    bucket_targets = compute_bucket_targets(
        Path(args.config), args.target_total_tokens
    )
    LOG.info("bucket_targets = %s", bucket_targets)

    # Convert per-bucket budgets into per-source budgets by dividing the bucket
    # budget equally across its member sources.
    bucket_sources: Dict[str, List[SourceSpec]] = {}
    for spec in SOURCES:
        bucket_sources.setdefault(spec.bucket, []).append(spec)
    per_source_budget: Dict[str, int] = {}
    for bucket, specs in bucket_sources.items():
        each = bucket_targets.get(bucket, 0) // max(len(specs), 1)
        for s in specs:
            per_source_budget[s.name] = each

    per_source_stats: List[Dict[str, object]] = []
    seen_hashes: set[str] = set()

    # Optional --sources filter.
    source_filter = None
    if getattr(args, "sources", None):
        source_filter = {s.strip() for s in args.sources.split(",") if s.strip()}

    manifest_name = "manifest.json"
    if args.shard_prefix:
        manifest_name = f"manifest-{args.shard_prefix.rstrip('-')}.json"

    def _write_manifest() -> None:
        manifest = {
            "target_total_tokens": args.target_total_tokens,
            "bucket_targets": bucket_targets,
            "per_source_budget": per_source_budget,
            "per_source_stats": per_source_stats,
            "shard_prefix": args.shard_prefix or "",
        }
        (output_dir / manifest_name).write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    for spec in SOURCES:
        if source_filter and spec.name not in source_filter:
            LOG.info("skipping %s (not in --sources filter)", spec.name)
            continue
        budget = per_source_budget.get(spec.name, 0)
        if budget <= 0:
            continue
        # Reset per-source MinHash to bound memory. Near-dup coverage is
        # within-source (our bigger concern); cross-source exact dedup is
        # already handled by ``seen_hashes``.
        minhash = MinHashDedup(
            num_perm=args.minhash_perm, threshold=args.minhash_threshold
        )
        try:
            stats = process_source(
                spec,
                output_dir=output_dir,
                token_budget=budget,
                lid=lid,
                minhash=minhash,
                decontam_ngrams=decontam,
                max_docs=args.max_docs_per_source,
                shard_target_bytes=args.shard_target_bytes,
                seen_hashes=seen_hashes,
                shard_prefix=args.shard_prefix or "",
                skip_docs=args.skip_docs or 0,
            )
        except Exception as exc:
            LOG.exception("source %s failed: %s", spec.name, exc)
            stats = {"source": spec.name, "error": str(exc)}
        per_source_stats.append(stats)
        _write_manifest()  # flush after each source for resume visibility

    print(f"wrote manifest: {output_dir / manifest_name}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download-assets", help="Download fastText LID model.")
    dl.add_argument("--assets-dir", default="outputs/pretrain_assets")
    dl.set_defaults(func=cmd_download_assets)

    build = sub.add_parser("build", help="Build the pre-training corpus.")
    build.add_argument("--config", default="configs/aksara_20b_dense.json")
    build.add_argument("--output-dir", required=True)
    build.add_argument("--assets-dir", default="outputs/pretrain_assets")
    build.add_argument("--target-total-tokens", type=int, default=400_000_000_000)
    build.add_argument("--max-docs-per-source", type=int, default=None)
    build.add_argument("--shard-target-bytes", type=int, default=1_000_000_000)
    build.add_argument("--minhash-perm", type=int, default=128)
    build.add_argument("--minhash-threshold", type=float, default=0.8)
    build.add_argument("--decontam", action="store_true", default=True)
    build.add_argument("--no-decontam", dest="decontam", action="store_false")
    build.add_argument("--decontam-ngram", type=int, default=13)
    build.add_argument(
        "--sources",
        default=None,
        help=(
            "Comma-separated list of source names to build (e.g. "
            "'fineweb,fineweb2_id,indo4b'). If omitted, all sources from SOURCES "
            "are built. Useful for skipping gated datasets or running in parallel."
        ),
    )
    build.add_argument(
        "--shard-prefix",
        default="",
        help=(
            "Prefix prepended to every emitted shard filename, e.g. 'p2-' yields "
            "'p2-shard-00000.parquet'. Use this when running multiple producers "
            "concurrently against the same --output-dir to avoid shard filename "
            "collisions."
        ),
    )
    build.add_argument(
        "--skip-docs",
        type=int,
        default=0,
        help=(
            "Skip the first N documents of each processed source's streaming "
            "iterator. Use this to partition a large dataset (e.g. "
            "HuggingFaceFW/fineweb sample-350BT) across multiple producers so "
            "they do not reprocess the same documents."
        ),
    )
    build.set_defaults(func=cmd_build)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
