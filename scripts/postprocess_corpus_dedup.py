"""Post-hoc cross-producer deduplication for the pre-training corpus.

The producer ``scripts/build_pretrain_corpus_v2.py`` runs multiple concurrent
workers against (partially) overlapping source streams (``fineweb sample-10BT``
vs ``sample-100BT`` vs ``sample-350BT``). Each producer only MinHash-dedups
*within its own process*, so cross-producer duplicates survive into the
``gs://…/pretrain/v1/`` shard set. This script sweeps the full Parquet shard
set once and emits a deduplicated mirror at a user-chosen destination.

Pipeline
========

1. Enumerate every ``*.parquet`` under ``--input-prefix`` (GCS URI).
2. For each doc:
   a. Exact-dedup via a rolling SHA-256 set.
   b. Near-dedup via a single MinHash-LSH index over the whole corpus.
3. Write surviving docs to ``--output-prefix`` (GCS URI) as Parquet shards
   named ``dedup-shard-NNNNN.parquet`` of roughly ``--shard-target-bytes`` each.
4. Emit ``dedup_manifest.json`` summarizing before/after counts & token counts.

Memory notes
------------

- The exact-dedup set holds one SHA-256 hex digest per document. For 500 M
  documents this is ≈32 GB. v6e-8 has 1.4 TB so that is comfortably in budget.
- MinHash-LSH stores ``num_perm`` hashes per doc (32 bytes each). For
  ``num_perm=128`` and 500 M docs: ~64 GB. Still within budget.

CLI example
-----------

    python3 scripts/postprocess_corpus_dedup.py \\
        --input-prefix gs://aksarallm20b-eu/pretrain/v1/ \\
        --output-prefix gs://aksarallm20b-eu/pretrain/v1-dedup/ \\
        --shard-target-bytes 536870912 \\
        --num-perm 128 \\
        --threshold 0.8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Iterator, List, Optional

LOG = logging.getLogger("dedup")


# ---------------------------------------------------------------------------
# MinHash helpers (small vendored subset to keep this script self-contained)
# ---------------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    # Match the producer's estimate in build_pretrain_corpus_v2.py:
    # 0.75 * whitespace word count. Keeping these formulas in sync
    # ensures ``dedup_manifest.json`` token counts are comparable to
    # the per-producer manifests for downstream data-mix accounting.
    return max(1, int(0.75 * len(text.split())))


class MinHashLSHIndex:
    """Thin wrapper over ``datasketch.MinHash`` + ``MinHashLSH``."""

    def __init__(self, num_perm: int, threshold: float) -> None:
        from datasketch import MinHash, MinHashLSH  # noqa: import-time

        self._MinHash = MinHash
        self._num_perm = num_perm
        self._lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self._counter = 0

    def insert_if_new(self, text: str) -> bool:
        mh = self._MinHash(num_perm=self._num_perm)
        # 5-gram shingles at word level; cheap and good enough for dedup.
        toks = text.split()
        for i in range(max(len(toks) - 4, 1)):
            mh.update(" ".join(toks[i : i + 5]).encode("utf-8", errors="ignore"))
        hits = self._lsh.query(mh)
        if hits:
            return False
        key = f"d{self._counter}"
        self._counter += 1
        self._lsh.insert(key, mh)
        return True


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


# ---------------------------------------------------------------------------
# GCS I/O
# ---------------------------------------------------------------------------


def _list_parquet(fs, prefix: str) -> List[str]:
    assert prefix.startswith("gs://")
    entries = fs.find(prefix.rstrip("/") + "/")
    out = [e for e in entries if e.endswith(".parquet")]
    out.sort()
    return out


def _stream_docs(fs, path: str) -> Iterator[dict]:
    import pyarrow.parquet as pq

    with fs.open(path, "rb") as f:
        tbl = pq.read_table(f)
    # Yield small python dicts row-by-row to minimise peak RSS.
    for row in tbl.to_pylist():
        yield row


class ShardWriter:
    def __init__(self, fs, output_prefix: str, shard_target_bytes: int) -> None:
        self._fs = fs
        self._prefix = output_prefix.rstrip("/") + "/"
        self._target = shard_target_bytes
        self._buf: List[dict] = []
        self._buf_bytes = 0
        self._shard = 0

    def write(self, row: dict) -> None:
        self._buf.append(row)
        self._buf_bytes += len(row.get("text", "") or "")
        if self._buf_bytes >= self._target:
            self._flush()

    def _flush(self) -> None:
        if not self._buf:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        name = f"dedup-shard-{self._shard:05d}.parquet"
        path = self._prefix + name
        tbl = pa.Table.from_pylist(self._buf)
        with self._fs.open(path, "wb") as f:
            pq.write_table(tbl, f, compression="zstd")
        LOG.info("wrote %s (%d docs, %.1f MB)", path, len(self._buf), self._buf_bytes / 1e6)
        self._shard += 1
        self._buf.clear()
        self._buf_bytes = 0

    def close(self) -> None:
        self._flush()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input-prefix", required=True, help="gs:// URI containing raw shards")
    p.add_argument("--output-prefix", required=True, help="gs:// URI to write dedup shards")
    p.add_argument("--shard-target-bytes", type=int, default=536_870_912, help="≈512 MB")
    p.add_argument("--num-perm", type=int, default=128)
    p.add_argument("--threshold", type=float, default=0.8)
    p.add_argument("--log-every", type=int, default=10_000)
    p.add_argument(
        "--exact-only",
        action="store_true",
        help="Skip MinHash-LSH (much faster; use to debug or when you only want SHA-exact dedup).",
    )
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import gcsfs

    fs = gcsfs.GCSFileSystem()

    paths = _list_parquet(fs, args.input_prefix)
    LOG.info("found %d parquet shards under %s", len(paths), args.input_prefix)
    if not paths:
        return 1

    seen: set[str] = set()
    lsh: Optional[MinHashLSHIndex] = None
    if not args.exact_only:
        lsh = MinHashLSHIndex(num_perm=args.num_perm, threshold=args.threshold)

    writer = ShardWriter(fs, args.output_prefix, args.shard_target_bytes)

    stats = {
        "input_shards": len(paths),
        "seen_docs": 0,
        "kept_docs": 0,
        "kept_tokens": 0,
        "dropped_exact": 0,
        "dropped_near": 0,
        "dropped_empty": 0,
    }

    t0 = time.time()
    for shard_idx, path in enumerate(paths, start=1):
        LOG.info("[%d/%d] scanning %s", shard_idx, len(paths), path)
        try:
            for row in _stream_docs(fs, path):
                stats["seen_docs"] += 1
                text = row.get("text") or ""
                if not isinstance(text, str) or not text.strip():
                    stats["dropped_empty"] += 1
                    continue
                h = _sha(text)
                if h in seen:
                    stats["dropped_exact"] += 1
                    continue
                seen.add(h)
                if lsh is not None and not lsh.insert_if_new(text):
                    stats["dropped_near"] += 1
                    continue
                stats["kept_docs"] += 1
                stats["kept_tokens"] += _estimate_tokens(text)
                writer.write(row)
                if stats["seen_docs"] % args.log_every == 0:
                    LOG.info(
                        "progress: seen=%d kept=%d (%.1f M tok, %.1f%% keep) in %.1fs",
                        stats["seen_docs"],
                        stats["kept_docs"],
                        stats["kept_tokens"] / 1e6,
                        100.0 * stats["kept_docs"] / max(1, stats["seen_docs"]),
                        time.time() - t0,
                    )
        except Exception as exc:
            LOG.exception("shard %s failed: %s; continuing", path, exc)

    writer.close()

    manifest_path = args.output_prefix.rstrip("/") + "/dedup_manifest.json"
    with fs.open(manifest_path, "wb") as f:
        f.write(json.dumps(stats, ensure_ascii=False, indent=2).encode("utf-8"))
    LOG.info("wrote manifest: %s", manifest_path)
    LOG.info("DONE: %s", json.dumps(stats, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
