#!/usr/bin/env python3
"""Train the AksaraLLM 20B byte-level BPE tokenizer.

Target: 131,072-slot vocabulary matching ``configs/aksara_20b_dense.json``.

Design choices (all locked by this script — do not diverge without a config bump):

* **Byte-level pre-tokenizer** (GPT-2/LLaMA-3 style). Guarantees full byte
  coverage so any byte sequence, including Indonesian local scripts or
  emoji, can be encoded losslessly.
* **BPE model.** Standard choice for multilingual LLaMA-class models;
  matches the Qwen2/LLaMA-3 ecosystem and is supported natively by
  ``tokenizers``, llama.cpp, and vLLM.
* **Reserved tail of 256 slots** (IDs 130816..131071) for future special
  tokens, tool tokens, FIM markers, etc. Training vocab is therefore
  131072 - 256 - len(NAMED_SPECIAL_TOKENS) = 130800 learnable merges.
* **Special token set** committed at training time and never re-ordered:

      <|pad|>, <|bos|>, <|eos|>, <|unk|>,
      <|system|>, <|user|>, <|assistant|>, <|tool|>,
      <|im_start|>, <|im_end|>,
      <|fim_prefix|>, <|fim_middle|>, <|fim_suffix|>,
      <|endoftext|>

  IDs 0..13 are pinned in that order so checkpoints produced during
  pretraining stay valid under the same special-token layout.

Usage
-----

The heavy lifting is corpus preparation. This script expects a plain-text
file (or a directory of ``.txt``/``.jsonl`` files, one document per line
with a ``text`` field for JSONL) containing a balanced sample of
English web, multilingual web, Indonesian targeted corpora, and code.
The blueprint's target mix for the tokenizer training sample is
roughly 40 / 25 / 20 / 15 (en / multilingual / id+jv+su / code) on
~50–100 GB of raw text.

Typical invocation::

    python scripts/train_tokenizer_20b.py \
        --corpus /data/tokenizer_corpus/ \
        --output-dir outputs/tokenizer-20b \
        --vocab-size 131072 \
        --min-frequency 2 \
        --max-bytes $((80 * 1024 ** 3))

Fertility measurement (on a held-out set you curate yourself)::

    python scripts/train_tokenizer_20b.py measure-fertility \
        --tokenizer outputs/tokenizer-20b/tokenizer.json \
        --sample-id path/to/held_out_id.txt \
        --sample-en path/to/held_out_en.txt \
        --sample-code path/to/held_out_code.txt
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence

# Special tokens with pinned IDs. The first entry gets id 0, the second 1,
# and so on; the training vocab slots start at len(NAMED_SPECIAL_TOKENS).
NAMED_SPECIAL_TOKENS: Sequence[str] = (
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|unk|>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|fim_prefix|>",
    "<|fim_middle|>",
    "<|fim_suffix|>",
    "<|endoftext|>",
)

# Number of empty slots reserved at the end of the vocab for future special
# tokens that we do not yet want to commit to. These slots are created as
# ``<|reserved_N|>`` added-tokens so they are never emitted by the BPE
# model but are guaranteed not to collide with future additions.
RESERVED_TAIL = 256


# ---------------------------------------------------------------------------
# Corpus iteration
# ---------------------------------------------------------------------------


def _iter_jsonl(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get("text") or obj.get("content") or ""
            if text:
                yield text


def _iter_plain(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                yield line


def iter_corpus(corpus: Path, max_bytes: int | None = None) -> Iterator[str]:
    """Yield one document at a time from a file or directory.

    Supports ``.jsonl`` (expects a ``text`` or ``content`` field),
    ``.txt``, ``.text`` and plain files. If ``corpus`` is a directory,
    every supported file under it (non-recursive, then recursive as a
    fallback) is streamed in shuffled order so training doesn't
    over-index on one source.
    """

    paths: List[Path]
    if corpus.is_file():
        paths = [corpus]
    elif corpus.is_dir():
        # Always walk recursively and keep only recognised shard extensions.
        # ``manifest.json`` and other bookkeeping files must not be fed to
        # the BPE trainer (it silently trains on them and you end up with a
        # 586-slot vocab). Matching suffixes here is the load-bearing fix.
        allowed = {".txt", ".text", ".jsonl"}
        paths = [
            p for p in corpus.rglob("*")
            if p.is_file() and p.suffix.lower() in allowed
        ]
        random.shuffle(paths)
    else:
        raise FileNotFoundError(f"corpus path not found: {corpus}")

    if not paths:
        raise FileNotFoundError(
            f"no .txt/.text/.jsonl files found under {corpus}. "
            "Populate the corpus directory first."
        )

    seen_bytes = 0
    for path in paths:
        suffix = path.suffix.lower()
        stream: Iterator[str]
        if suffix == ".jsonl":
            stream = _iter_jsonl(path)
        elif suffix in {".txt", ".text", ""}:
            stream = _iter_plain(path)
        else:
            # Unknown extension: best-effort plain text.
            stream = _iter_plain(path)

        for text in stream:
            yield text
            if max_bytes is not None:
                seen_bytes += len(text.encode("utf-8", errors="ignore"))
                if seen_bytes >= max_bytes:
                    return


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _build_trainer_and_tokenizer(vocab_size: int, min_frequency: int):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    special_tokens = list(NAMED_SPECIAL_TOKENS)
    special_tokens += [f"<|reserved_{i}|>" for i in range(RESERVED_TAIL)]

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    return tokenizer, trainer, special_tokens


def _write_tokenizer_config(output_dir: Path, vocab_size: int) -> None:
    """Write a Hugging Face ``tokenizer_config.json`` for ``AutoTokenizer``."""

    cfg = {
        "add_bos_token": False,
        "add_eos_token": False,
        "added_tokens_decoder": {},
        "bos_token": "<|bos|>",
        "clean_up_tokenization_spaces": False,
        "eos_token": "<|eos|>",
        "model_max_length": 131072,
        "pad_token": "<|pad|>",
        "tokenizer_class": "PreTrainedTokenizerFast",
        "unk_token": "<|unk|>",
        "chat_template": (
            "{% for message in messages %}"
            "{{ '<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n' }}"
            "{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|im_start|>assistant\\n' }}{% endif %}"
        ),
    }
    (output_dir / "tokenizer_config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Minimal special_tokens_map.json for tools that look there.
    special_map = {
        "bos_token": "<|bos|>",
        "eos_token": "<|eos|>",
        "pad_token": "<|pad|>",
        "unk_token": "<|unk|>",
    }
    (output_dir / "special_tokens_map.json").write_text(
        json.dumps(special_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # A short README so humans inspecting the repo understand the layout.
    readme = f"""# AksaraLLM 20B tokenizer

Byte-level BPE, `vocab_size={vocab_size}`, produced by
`scripts/train_tokenizer_20b.py`.

Special tokens (IDs 0..{len(NAMED_SPECIAL_TOKENS) - 1} in order):

{chr(10).join(f"- `{i}` → `{t}`" for i, t in enumerate(NAMED_SPECIAL_TOKENS))}

The last {RESERVED_TAIL} IDs are reserved as `<|reserved_N|>` for future
expansion without breaking already-pretrained checkpoints.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def cmd_train(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus = Path(args.corpus)

    tokenizer, trainer, special_tokens = _build_trainer_and_tokenizer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )

    t0 = time.time()
    print(
        f"[train_tokenizer_20b] corpus={corpus}, "
        f"vocab_size={args.vocab_size}, "
        f"specials={len(special_tokens)} (named={len(NAMED_SPECIAL_TOKENS)}, reserved={RESERVED_TAIL}), "
        f"max_bytes={args.max_bytes}",
        flush=True,
    )

    def stream() -> Iterable[str]:
        for i, text in enumerate(iter_corpus(corpus, max_bytes=args.max_bytes)):
            if i and i % 1_000_000 == 0:
                elapsed = time.time() - t0
                print(
                    f"  streamed {i:,} docs, {elapsed:.0f}s elapsed",
                    flush=True,
                )
            yield text

    tokenizer.train_from_iterator(stream(), trainer=trainer)
    tokenizer.save(str(output_dir / "tokenizer.json"))
    _write_tokenizer_config(output_dir, args.vocab_size)

    final_vocab = tokenizer.get_vocab_size()
    print(
        f"[train_tokenizer_20b] done in {time.time() - t0:.1f}s, "
        f"final vocab size = {final_vocab}",
        flush=True,
    )
    if final_vocab != args.vocab_size:
        # Training can undershoot vocab_size when min_frequency is too high
        # for the corpus. Warn loudly — the 20B model config expects exactly
        # 131072 slots because the embedding table size is baked in.
        print(
            f"[train_tokenizer_20b] WARNING: final vocab ({final_vocab}) "
            f"!= target ({args.vocab_size}). Fix the corpus or --min-frequency "
            "and retrain. Do NOT pretrain on an off-size tokenizer.",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Fertility measurement
# ---------------------------------------------------------------------------


def _fertility(tokenizer, text: str) -> float:
    """Return tokens-per-word ratio on ``text``."""

    words = text.split()
    if not words:
        return 0.0
    ids = tokenizer.encode(text).ids
    return len(ids) / len(words)


def cmd_measure_fertility(args: argparse.Namespace) -> None:
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(args.tokenizer))

    samples = {
        "id": args.sample_id,
        "en": args.sample_en,
        "code": args.sample_code,
    }
    results = {}
    for lang, path in samples.items():
        if path is None:
            continue
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
        results[lang] = _fertility(tok, text)

    for lang, ratio in results.items():
        target = "<= 1.6" if lang == "id" else "<= 1.4" if lang == "en" else "<= 2.0"
        print(f"fertility[{lang}] = {ratio:.3f} (target {target})")

    if "id" in results and results["id"] > 1.8:
        print("WARNING: Indonesian fertility > 1.8 — too fragmented. Retrain with more id data.")
    if "en" in results and results["en"] > 1.6:
        print("WARNING: English fertility > 1.6 — too fragmented. Usually means en sample is too small.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=False)

    train = sub.add_parser("train", help="Train a new BPE tokenizer (default command).")
    train.add_argument("--corpus", required=True, help="File or directory with training text.")
    train.add_argument("--output-dir", required=True, help="Where to write tokenizer.json + config.")
    train.add_argument("--vocab-size", type=int, default=131072)
    train.add_argument("--min-frequency", type=int, default=2)
    train.add_argument(
        "--max-bytes",
        type=int,
        default=None,
        help="Hard cap on raw UTF-8 bytes streamed from the corpus. "
        "Use for reproducibility / smoke tests.",
    )
    train.set_defaults(func=cmd_train)

    fert = sub.add_parser("measure-fertility", help="Print tokens-per-word on held-out samples.")
    fert.add_argument("--tokenizer", required=True, help="Path to tokenizer.json.")
    fert.add_argument("--sample-id", default=None)
    fert.add_argument("--sample-en", default=None)
    fert.add_argument("--sample-code", default=None)
    fert.set_defaults(func=cmd_measure_fertility)

    # Bare invocation (no subcommand): default to train. Keeps the CLI
    # compatible with simple shell examples.
    parser.set_defaults(func=cmd_train)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # When used without a subcommand, argparse won't attach --corpus/etc.
    # to the namespace. Re-run with 'train' prepended so the user gets a
    # clean error message instead of an AttributeError.
    if args.command is None and not hasattr(args, "corpus"):
        parser.error("Provide 'train' or 'measure-fertility' as the first argument.")

    random.seed(1337)
    args.func(args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
