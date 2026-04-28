"""Pre-tokenize a deduplicated parquet corpus into packed uint32 sequence shards.

Reads parquet shards under --input-prefix (each row has a 'text' column),
encodes each document with a HuggingFace tokenizer, prepends/separates with
BOS/EOS, packs the concatenated stream into fixed-length sequences, and writes
them out as ``.npy`` files (shape ``[N_seqs, seq_len]`` dtype ``uint32``).

Designed to run as multiple parallel workers, one per partition (see
``--shard-range`` for picking a slice of the input shard list). Token counts
are aggregated into a per-worker manifest so the global BPE token count is the
sum of all worker manifests.

Usage (one worker over a single partition)::

    python -u scripts/tokenize_pretrain_corpus.py \\
        --input-prefix gs://aksarallm20b-eu/pretrain/v1-dedup-v3-near/part0/ \\
        --output-prefix gs://aksarallm20b-eu/pretrain/v1-tokenized/part0/ \\
        --tokenizer Ezekiel999/aksara-tokenizer-20b \\
        --seq-len 8192 --seqs-per-shard 4096

Multiple workers can run concurrently against disjoint partitions; outputs
are independent and the per-partition manifests can be summed afterwards.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
from typing import Iterable, Iterator, List, Optional

LOG = logging.getLogger("tokenize")


def _list_parquet(fs, prefix: str) -> List[str]:
    if not prefix.endswith("/"):
        prefix = prefix + "/"
    paths = fs.glob(prefix + "*.parquet")
    return sorted(paths)


def _stream_docs(fs, path: str) -> Iterator[str]:
    import pyarrow.parquet as pq
    with fs.open(path, "rb") as f:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=1024, columns=["text"]):
            for x in batch.column("text"):
                t = x.as_py()
                if t:
                    yield t


def _resolve_special(tok, name: str) -> Optional[int]:
    """Best-effort lookup for BOS/EOS across tokenizers with varying token names.

    Slow (non-fast) tokenizers map unknown strings to ``unk_token_id`` rather than
    raising or returning ``None``, so we explicitly reject that ID when scanning
    candidates — otherwise we'd silently use UNK as the document separator.
    """
    cands = []
    if name == "bos":
        cands = ["<|bos|>", "<s>", "<|begin_of_text|>", "<|startoftext|>", "[BOS]"]
        if tok.bos_token_id is not None:
            return int(tok.bos_token_id)
    else:
        cands = ["<|eos|>", "</s>", "<|end_of_text|>", "<|endoftext|>", "[EOS]"]
        if tok.eos_token_id is not None:
            return int(tok.eos_token_id)
    unk_id = getattr(tok, "unk_token_id", None)
    for c in cands:
        try:
            tid = tok.convert_tokens_to_ids(c)
            if tid is not None and tid >= 0 and tid != unk_id:
                return int(tid)
        except Exception:
            pass
    return None


def _flush_shard(fs, out_prefix: str, shard_idx: int, seqs: list, dtype) -> int:
    import numpy as np
    arr = np.asarray(seqs, dtype=dtype)
    rel = f"seq-{shard_idx:05d}.npy"
    if not out_prefix.endswith("/"):
        out_prefix = out_prefix + "/"
    out_path = out_prefix + rel
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    buf.seek(0)
    with fs.open(out_path, "wb") as fh:
        fh.write(buf.getvalue())
    return arr.size  # number of tokens written in this shard


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input-prefix", required=True,
                   help="GCS prefix containing dedup-shard-*.parquet inputs.")
    p.add_argument("--output-prefix", required=True,
                   help="GCS prefix to write seq-*.npy outputs to.")
    p.add_argument("--tokenizer", required=True,
                   help="HF tokenizer name or local path.")
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--seqs-per-shard", type=int, default=4096,
                   help="N sequences per .npy file (default 4096; ~134 MB at seq_len=8192 uint32).")
    p.add_argument("--shard-range", type=str, default=None,
                   help="Slice of input shards to process, e.g. '0:80'. Used for parallel workers.")
    p.add_argument("--log-every", type=int, default=10_000)
    p.add_argument("--max-shards", type=int, default=None,
                   help="Stop after this many output shards (debug).")
    p.add_argument("--dtype", choices=["uint16", "uint32"], default="uint32")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    import gcsfs  # type: ignore
    import numpy as np  # type: ignore
    from transformers import AutoTokenizer  # type: ignore

    fs = gcsfs.GCSFileSystem()
    paths = _list_parquet(fs, args.input_prefix)
    LOG.info("found %d parquet shards under %s", len(paths), args.input_prefix)
    if not paths:
        return 1

    if args.shard_range:
        try:
            a, b = args.shard_range.split(":")
            paths = paths[(int(a) if a else 0) : (int(b) if b else len(paths))]
        except ValueError:
            LOG.error("--shard-range must be 'START:END'")
            return 2
        LOG.info("shard-range %s -> processing %d shards", args.shard_range, len(paths))

    LOG.info("loading tokenizer: %s", args.tokenizer)
    tok = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    bos_id = _resolve_special(tok, "bos")
    eos_id = _resolve_special(tok, "eos")
    if eos_id is None:
        LOG.error("no EOS token found in tokenizer; cannot pack documents safely")
        return 3
    LOG.info("vocab_size=%d bos=%s eos=%s", tok.vocab_size, bos_id, eos_id)

    dtype = np.uint16 if args.dtype == "uint16" else np.uint32
    # ``len(tok)`` includes added/special tokens which sit above ``vocab_size``;
    # uint16 max ID is 65535 so the limit is ``len(tok) > 65536``.
    total_vocab = len(tok)
    if dtype is np.uint16 and total_vocab > 65536:
        LOG.error("tokenizer vocab %d (incl. added tokens) exceeds uint16; pass --dtype uint32",
                  total_vocab)
        return 4

    seq_len = args.seq_len
    seqs_per_shard = args.seqs_per_shard
    # If we prepend BOS after every flush, each pack iteration consumes
    # ``seq_len - 1`` net tokens from the buffer, so seq_len must be >= 2 to
    # make forward progress. Without BOS the floor is 1 (still useless but
    # not a hang).
    min_seq_len = 2 if bos_id is not None else 1
    if seq_len < min_seq_len:
        LOG.error("seq_len=%d too small (need >=%d when bos_id=%s)", seq_len, min_seq_len, bos_id)
        return 5
    buf: list[int] = []
    if bos_id is not None:
        buf.append(bos_id)

    out_seqs: list[list[int]] = []
    shard_idx = 0
    stats = {
        "input_shards_processed": 0,
        "input_docs": 0,
        "output_sequences": 0,
        "output_tokens": 0,
        "output_shards_written": 0,
    }
    t0 = time.time()
    last_log = 0
    for shard_path in paths:
        LOG.info("[%d/%d] %s", stats["input_shards_processed"] + 1, len(paths), shard_path)
        for text in _stream_docs(fs, shard_path):
            stats["input_docs"] += 1
            ids = tok(text, add_special_tokens=False, truncation=False)["input_ids"]
            buf.extend(ids)
            buf.append(eos_id)
            while len(buf) >= seq_len:
                seq = buf[:seq_len]
                rest = buf[seq_len:]
                buf = ([bos_id] if bos_id is not None else []) + rest
                out_seqs.append(seq)
                stats["output_sequences"] += 1
                if len(out_seqs) >= seqs_per_shard:
                    written = _flush_shard(fs, args.output_prefix, shard_idx, out_seqs, dtype)
                    stats["output_tokens"] += written
                    stats["output_shards_written"] += 1
                    shard_idx += 1
                    out_seqs = []
                    if args.max_shards is not None and shard_idx >= args.max_shards:
                        LOG.info("max-shards reached, stopping early")
                        return _finalize(fs, args, stats, t0)
            if stats["input_docs"] - last_log >= args.log_every:
                last_log = stats["input_docs"]
                rate = stats["input_docs"] / max(1.0, time.time() - t0)
                LOG.info(
                    "progress: docs=%d seqs=%d shards=%d (%.0f docs/s)",
                    stats["input_docs"], stats["output_sequences"],
                    stats["output_shards_written"], rate,
                )
        stats["input_shards_processed"] += 1

    if out_seqs:
        written = _flush_shard(fs, args.output_prefix, shard_idx, out_seqs, dtype)
        stats["output_tokens"] += written
        stats["output_shards_written"] += 1
        shard_idx += 1

    return _finalize(fs, args, stats, t0)


def _finalize(fs, args, stats, t0) -> int:
    elapsed = time.time() - t0
    stats["elapsed_seconds"] = elapsed
    stats["seq_len"] = args.seq_len
    stats["seqs_per_shard"] = args.seqs_per_shard
    stats["tokenizer"] = args.tokenizer
    stats["dtype"] = args.dtype
    stats["input_prefix"] = args.input_prefix
    stats["output_prefix"] = args.output_prefix
    out_prefix = args.output_prefix
    if not out_prefix.endswith("/"):
        out_prefix += "/"
    manifest_path = out_prefix + "tokenize_manifest.json"
    with fs.open(manifest_path, "wt") as f:
        json.dump(stats, f, indent=2)
    LOG.info("wrote manifest: %s", manifest_path)
    LOG.info("FINAL %s", json.dumps({k: stats[k] for k in
                                     ("input_shards_processed", "input_docs",
                                      "output_sequences", "output_tokens",
                                      "output_shards_written", "elapsed_seconds")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
