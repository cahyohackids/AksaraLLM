#!/usr/bin/env python3
"""
Unified dataset builder for AksaraLLM.

Outputs:
  - aksara_sft_base.jsonl
  - aksara_sft_synthetic.jsonl
  - aksara_sft_final.jsonl
  - aksara_dpo.jsonl
  - aksara_sigap.jsonl
  - aksara_pretrain_seed.jsonl
  - dataset_manifest.json

This script prefers local/cached data first, then optionally expands the
dataset with an open-source teacher model through Hugging Face Inference.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import logging
import os
import random
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator


LOG = logging.getLogger("aksara-datasets")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = REPO_ROOT.parent
OUTPUT_DIR = REPO_ROOT / "data" / "generated"

LOCAL_SFT_JSONL = WORKSPACE_ROOT / "aksarallm_v3_data.jsonl"
LOCAL_SIGAP_JSON = WORKSPACE_ROOT / "sigap_dataset_clean_v2.json"
LOCAL_SIGAP_AUG_JSON = WORKSPACE_ROOT / "sigap_dataset_clean_augmented_v2.json"
LOCAL_SIGAP_COMPLEX_JSON = WORKSPACE_ROOT / "sigap_dataset_complex_scenarios_v1.json"
LOCAL_SIGAP_PY = WORKSPACE_ROOT / "aksarallm_sigap_v5.py"
TUMPENG_CACHE_GLOB = (
    WORKSPACE_ROOT
    / ".cache"
    / "huggingface"
    / "datasets"
    / "afrizalha___tumpeng-1-indonesian"
    / "default"
    / "0.0.0"
)

SPACE_RE = re.compile(r"[ \t]+")
BLANK_RE = re.compile(r"\n{3,}")
ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200d\ufeff]")
NON_WORD_RE = re.compile(r"[^a-z0-9\s]")
REPEATED_CHAR_RE = re.compile(r"(.)\1{6,}")

INDO_HINTS = {
    "yang", "dan", "di", "ke", "dari", "untuk", "pada", "adalah", "dengan",
    "tidak", "akan", "bisa", "sudah", "juga", "lebih", "karena", "dalam",
    "sebagai", "jika", "atau", "kami", "kita", "anda", "kamu",
}

BAD_CONTENT_RE = re.compile(
    r"\b(judi|slot|casino|porno|bokep|phishing|scam|narkoba|terorisme)\b",
    re.IGNORECASE,
)
WRONG_IDENTITY_RE = re.compile(
    r"\b(saya adalah|i am|i'm)\s+(chatgpt|gpt-4|gpt|gemini|claude|openai)\b",
    re.IGNORECASE,
)
META_RE = re.compile(
    r"(as an ai language model|sebagai model bahasa|openai|assistant created by openai)",
    re.IGNORECASE,
)

SYNTHETIC_CATEGORY_PROMPTS = [
    {
        "name": "identity",
        "prompt": (
            "Buat 5 pasangan instruction-response SFT tentang identitas AksaraLLM. "
            "Aturan: AksaraLLM adalah AI Indonesia, bukan ChatGPT, bukan Gemini, "
            "bukan Claude. Jawaban harus ramah, tegas, dan seluruhnya Bahasa Indonesia."
        ),
    },
    {
        "name": "safety",
        "prompt": (
            "Buat 5 pasangan instruction-response SFT untuk topik safety. "
            "Instruction berisi permintaan berbahaya seperti bom, hacking, penipuan, "
            "doxxing, malware, atau kekerasan. Response harus menolak dengan sopan, "
            "menjelaskan alasannya singkat, dan mengalihkan ke alternatif aman."
        ),
    },
    {
        "name": "reasoning",
        "prompt": (
            "Buat 5 pasangan instruction-response SFT tentang reasoning dan analisis. "
            "Gunakan Bahasa Indonesia. Variasikan antara logika, sebab-akibat, "
            "perbandingan, pengurutan kronologis, dan pengambilan keputusan."
        ),
    },
    {
        "name": "coding",
        "prompt": (
            "Buat 5 pasangan instruction-response SFT tentang coding Python atau web dasar. "
            "Jawaban harus jelas, praktis, dan boleh memuat blok kode singkat."
        ),
    },
    {
        "name": "matematika",
        "prompt": (
            "Buat 5 pasangan instruction-response SFT tentang matematika dasar hingga menengah "
            "dalam Bahasa Indonesia. Sertakan langkah penyelesaian singkat."
        ),
    },
    {
        "name": "indonesia",
        "prompt": (
            "Buat 5 pasangan instruction-response SFT tentang sejarah, budaya, geografi, "
            "atau kebijakan publik Indonesia. Jawaban harus akurat dan informatif."
        ),
    },
    {
        "name": "terjemahan",
        "prompt": (
            "Buat 5 pasangan instruction-response SFT tentang penerjemahan antara Bahasa "
            "Indonesia, Inggris, dan bahasa daerah Indonesia. Output tetap seluruhnya "
            "dijelaskan dalam Bahasa Indonesia."
        ),
    },
    {
        "name": "penulisan",
        "prompt": (
            "Buat 5 pasangan instruction-response SFT tentang menulis ringkasan, email, "
            "esai pendek, atau perbaikan kalimat dalam Bahasa Indonesia."
        ),
    },
]


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = ZERO_WIDTH_RE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = SPACE_RE.sub(" ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = BLANK_RE.sub("\n\n", text)
    return text.strip()


def simplify_text(text: str) -> str:
    text = normalize_text(text).lower()
    text = NON_WORD_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def is_likely_code(text: str) -> bool:
    lowered = text.lower()
    markers = ("```", "def ", "class ", "import ", "return ", "console.log", "<html")
    return any(marker in lowered for marker in markers)


def indonesian_score(text: str) -> int:
    return len(set(simplify_text(text).split()) & INDO_HINTS)


def quality_score(instruction: str, response: str) -> float:
    score = 0.0
    if len(instruction) >= 10:
        score += 0.15
    if len(instruction) >= 25:
        score += 0.10
    if len(response) >= 40:
        score += 0.20
    if len(response) >= 120:
        score += 0.20
    if len(response) <= 2500:
        score += 0.10
    if "\n" in response:
        score += 0.10
    if any(ch in response for ch in ".:;"):
        score += 0.05
    if indonesian_score(response) >= 2 or is_likely_code(response):
        score += 0.10
    if REPEATED_CHAR_RE.search(response):
        score -= 0.25
    return max(0.0, min(score, 1.0))


def reject_reason(instruction: str, response: str, source: str) -> str | None:
    combined = f"{instruction}\n{response}"
    if len(instruction) < 8:
        return "instruction_too_short"
    if len(response) < 20:
        return "response_too_short"
    if len(response) > 3200:
        return "response_too_long"
    if BAD_CONTENT_RE.search(combined):
        return "bad_content"
    if WRONG_IDENTITY_RE.search(response):
        return "wrong_identity"
    if META_RE.search(response):
        return "meta_response"
    if REPEATED_CHAR_RE.search(combined):
        return "noisy_repetition"
    if not is_likely_code(response) and indonesian_score(response) < 1 and source != "synthetic":
        return "not_indonesian_enough"
    if quality_score(instruction, response) < 0.42:
        return "low_quality"
    return None


def deduplicate_records(records: Iterable[dict]) -> list[dict]:
    seen_exact: set[str] = set()
    seen_fuzzy: set[str] = set()
    unique: list[dict] = []
    for row in records:
        inst = row["instruction"]
        resp = row["response"]
        exact_key = hashlib.sha1(
            f"{simplify_text(inst)}|{simplify_text(resp)}".encode("utf-8")
        ).hexdigest()
        if exact_key in seen_exact:
            continue
        seen_exact.add(exact_key)

        fuzzy_key = hashlib.sha1(
            f"{simplify_text(inst)[:160]}|{simplify_text(resp)[:220]}".encode("utf-8")
        ).hexdigest()
        if fuzzy_key in seen_fuzzy:
            continue
        seen_fuzzy.add(fuzzy_key)
        unique.append(row)
    return unique


def iter_jsonl_records(path: Path, source_name: str) -> Iterator[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            instruction = normalize_text(item.get("instruction", ""))
            response = normalize_text(item.get("output", "") or item.get("response", ""))
            category = normalize_text(item.get("category", "")) or "general"
            if instruction and response:
                yield {
                    "instruction": instruction,
                    "response": response,
                    "category": category,
                    "source": source_name,
                }


def iter_sigap_records(path: Path, source_name: str) -> Iterator[dict]:
    if not path.exists():
        return
    items = json.loads(path.read_text(encoding="utf-8"))
    for item in items:
        instruction = normalize_text(item.get("q", ""))
        response = normalize_text(item.get("a", ""))
        if instruction and response:
            yield {
                "instruction": instruction,
                "response": response,
                "category": "sigap",
                "source": source_name,
            }


def iter_python_dataset(path: Path, source_name: str) -> Iterator[dict]:
    if not path.exists():
        return
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "DATASET":
                dataset = ast.literal_eval(node.value)
                for item in dataset:
                    if not isinstance(item, dict):
                        continue
                    instruction = normalize_text(item.get("q", ""))
                    response = normalize_text(item.get("a", ""))
                    if instruction and response:
                        yield {
                            "instruction": instruction,
                            "response": response,
                            "category": "sigap",
                            "source": source_name,
                        }
                return


def iter_tumpeng_cache_records(limit: int | None = None) -> Iterator[dict]:
    try:
        from datasets import Dataset, load_dataset
    except Exception as exc:
        LOG.warning("datasets import failed for Tumpeng cache: %s", exc)
        return

    arrow_files = sorted(TUMPENG_CACHE_GLOB.glob("*/tumpeng-1-indonesian-train.arrow"))
    if arrow_files:
        ds = Dataset.from_file(str(arrow_files[0]))
    else:
        LOG.info("No local Tumpeng cache found, loading from Hugging Face")
        try:
            ds = load_dataset("afrizalha/Tumpeng-1-Indonesian", split="train")
        except Exception as exc:
            LOG.warning("Unable to load Tumpeng dataset: %s", exc)
            return

    count = 0
    for row in ds:
        messages = row.get("messages") or []
        if not isinstance(messages, list):
            continue
        instruction = ""
        response = ""
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).lower().strip()
            content = normalize_text(message.get("content", ""))
            if not content:
                continue
            if not instruction and role == "user":
                instruction = content
                continue
            if instruction and role == "assistant":
                response = content
                break
        if instruction and response:
            yield {
                "instruction": instruction,
                "response": response,
                "category": "tumpeng",
                "source": "tumpeng_cache",
            }
            count += 1
            if limit and count >= limit:
                break


def iter_pretrain_seed_documents(sft_records: Iterable[dict]) -> Iterator[dict]:
    seen: set[str] = set()

    def maybe_emit(text: str, source: str) -> Iterator[dict]:
        text = normalize_text(text)
        if len(text) < 120:
            return
        key = hashlib.sha1(simplify_text(text[:800]).encode("utf-8")).hexdigest()
        if key in seen:
            return
        seen.add(key)
        yield {"text": text, "source": source}

    for markdown_path in sorted(REPO_ROOT.glob("**/*.md")):
        try:
            yield from maybe_emit(markdown_path.read_text(encoding="utf-8"), markdown_path.name)
        except Exception:
            continue

    for code_path in sorted(REPO_ROOT.glob("**/*.py")):
        try:
            yield from maybe_emit(code_path.read_text(encoding="utf-8"), code_path.name)
        except Exception:
            continue

    for workspace_code in sorted(WORKSPACE_ROOT.glob("aksarallm*.py")):
        try:
            yield from maybe_emit(workspace_code.read_text(encoding="utf-8"), workspace_code.name)
        except Exception:
            continue

    for row in sft_records:
        text = f"{row['instruction']}\n\n{row['response']}"
        yield from maybe_emit(text, f"seed::{row['source']}")


def parse_json_payload(raw_text: str) -> list[dict]:
    raw_text = raw_text.strip()
    if "```" in raw_text:
        blocks = re.findall(r"```(?:json)?\s*(.*?)```", raw_text, flags=re.DOTALL | re.IGNORECASE)
        if blocks:
            raw_text = blocks[0].strip()

    candidates = [raw_text]
    obj_match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    arr_match = re.search(r"\[.*\]", raw_text, flags=re.DOTALL)
    if obj_match:
        candidates.append(obj_match.group(0))
    if arr_match:
        candidates.append(arr_match.group(0))

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except Exception:
            continue
        if isinstance(payload, dict):
            for key in ("items", "data", "pairs"):
                if isinstance(payload.get(key), list):
                    return payload[key]
                if isinstance(payload.get(key), dict):
                    return [payload[key]]
        if isinstance(payload, list):
            flattened: list[dict] = []
            for item in payload:
                if isinstance(item, dict) and len(item) == 1:
                    nested = next(iter(item.values()))
                    if isinstance(nested, dict):
                        flattened.append(nested)
                        continue
                if isinstance(item, dict):
                    flattened.append(item)
            return flattened
    return []


def synthetic_batch(
    client,
    model_name: str,
    category_name: str,
    category_prompt: str,
    temperature: float = 0.8,
) -> list[dict]:
    system_prompt = (
        "Kamu adalah pembuat dataset AksaraLLM. "
        "Tulis seluruh jawaban dalam Bahasa Indonesia kecuali potongan kode. "
        "Kembalikan JSON object dengan kunci items, berisi daftar 5 objek dengan "
        "instruction, response, dan category. Jangan menulis markdown."
    )
    user_prompt = (
        f"{category_prompt}\n\n"
        "Pastikan tiap pasangan unik, berkualitas, aman, dan natural. "
        f"Setiap item wajib memakai category='{category_name}'."
    )
    result = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=900,
        temperature=temperature,
    )
    raw_text = result.choices[0].message.content or ""
    items = parse_json_payload(raw_text)
    cleaned: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        instruction = normalize_text(item.get("instruction", ""))
        response = normalize_text(item.get("response", ""))
        if instruction and response:
            cleaned.append(
                {
                    "instruction": instruction,
                    "response": response,
                    "category": normalize_text(item.get("category", category_name)) or category_name,
                    "source": "synthetic",
                }
            )
    return cleaned


def generate_synthetic_records(
    output_path: Path,
    model_name: str,
    target_count: int,
    seed: int,
) -> list[dict]:
    if target_count <= 0:
        return []
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        LOG.warning("HF_TOKEN is unset, skip synthetic generation")
        return []

    try:
        from huggingface_hub import InferenceClient
    except Exception as exc:
        LOG.warning("huggingface_hub import failed, skip synthetic generation: %s", exc)
        return []

    random.seed(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict] = []
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    existing.append(json.loads(line))

    if len(existing) >= target_count:
        LOG.info("Synthetic target already met: %s >= %s", len(existing), target_count)
        return existing[:target_count]

    client = InferenceClient(model=model_name, token=hf_token)
    all_rows = existing[:]
    consecutive_failures = 0
    payment_blocked = False
    with output_path.open("a", encoding="utf-8") as handle:
        while len(all_rows) < target_count:
            spec = random.choice(SYNTHETIC_CATEGORY_PROMPTS)
            try:
                batch = synthetic_batch(
                    client=client,
                    model_name=model_name,
                    category_name=spec["name"],
                    category_prompt=spec["prompt"],
                    temperature=round(random.uniform(0.65, 0.95), 2),
                )
                consecutive_failures = 0
            except Exception as exc:
                LOG.warning("Synthetic batch failed: %s", exc)
                consecutive_failures += 1
                if "402" in str(exc) or "Payment Required" in str(exc):
                    payment_blocked = True
                if payment_blocked and consecutive_failures >= 3:
                    LOG.warning(
                        "Stopping synthetic expansion early because inference credits are exhausted. "
                        "Keeping %s accepted synthetic rows.",
                        len(all_rows),
                    )
                    break
                if consecutive_failures >= 12:
                    LOG.warning(
                        "Stopping synthetic expansion after %s consecutive failures. "
                        "Keeping %s accepted synthetic rows.",
                        consecutive_failures,
                        len(all_rows),
                    )
                    break
                time.sleep(2)
                continue

            accepted = 0
            for row in batch:
                reason = reject_reason(row["instruction"], row["response"], source="synthetic")
                if reason:
                    continue
                all_rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                accepted += 1
                if len(all_rows) >= target_count:
                    break
            LOG.info(
                "Synthetic progress: %s/%s (+%s from %s)",
                len(all_rows),
                target_count,
                accepted,
                spec["name"],
            )
            if accepted == 0:
                consecutive_failures += 1
                if payment_blocked and consecutive_failures >= 3:
                    LOG.warning(
                        "Stopping synthetic expansion because no new rows were accepted and credits "
                        "appear exhausted. Keeping %s accepted synthetic rows.",
                        len(all_rows),
                    )
                    break
            time.sleep(1)
    return all_rows[:target_count]


def build_dpo_pairs(sft_rows: list[dict], target_count: int, seed: int) -> list[dict]:
    random.seed(seed)
    strategies = [
        lambda c: "Maaf, saya kurang tahu.",
        lambda c: "Saya adalah ChatGPT buatan OpenAI.",
        lambda c: " ".join(c.split()[: max(5, len(c.split()) // 6)]) + "...",
        lambda c: (c.split(".")[0] + ". ") * 2 if "." in c else c[:60] + "...",
        lambda c: "Jawaban ini sengaja dibuat terlalu singkat dan kurang membantu.",
        lambda c: "I can only answer in English.",
    ]
    rows: list[dict] = []
    for item in sft_rows:
        chosen = item["response"]
        if len(chosen) < 40:
            continue
        rejected = random.choice(strategies)(chosen)
        if rejected == chosen or len(rejected) < 10:
            continue
        rows.append(
            {
                "prompt": item["instruction"],
                "chosen": chosen,
                "rejected": rejected,
                "source": item["source"],
                "category": item["category"],
            }
        )
        if len(rows) >= target_count:
            break
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_base_sft(limit_tumpeng: int | None, include_augmented_sigap: bool) -> tuple[list[dict], dict]:
    raw_rows: list[dict] = []
    rejected = Counter()

    for loader in (
        iter_jsonl_records(LOCAL_SFT_JSONL, "aksarallm_v3"),
        iter_sigap_records(LOCAL_SIGAP_JSON, "sigap_clean_v2"),
        iter_sigap_records(LOCAL_SIGAP_COMPLEX_JSON, "sigap_complex_scenarios_v1"),
        iter_python_dataset(LOCAL_SIGAP_PY, "sigap_v5_dataset"),
        iter_tumpeng_cache_records(limit=limit_tumpeng),
    ):
        for row in loader or []:
            reason = reject_reason(row["instruction"], row["response"], row["source"])
            if reason:
                rejected[reason] += 1
                continue
            raw_rows.append(row)

    if include_augmented_sigap:
        for row in iter_sigap_records(LOCAL_SIGAP_AUG_JSON, "sigap_augmented_v2") or []:
            reason = reject_reason(row["instruction"], row["response"], row["source"])
            if reason:
                rejected[reason] += 1
                continue
            raw_rows.append(row)

    deduped = deduplicate_records(raw_rows)
    stats = {
        "raw_count": len(raw_rows),
        "final_count": len(deduped),
        "rejected": dict(rejected),
        "sources": dict(Counter(row["source"] for row in deduped)),
        "categories": dict(Counter(row["category"] for row in deduped)),
    }
    return deduped, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build unified AksaraLLM datasets.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--synthetic-target", type=int, default=0)
    parser.add_argument("--synthetic-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--dpo-target", type=int, default=4000)
    parser.add_argument("--limit-tumpeng", type=int, default=None)
    parser.add_argument("--include-augmented-sigap", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_sft, base_stats = build_base_sft(
        limit_tumpeng=args.limit_tumpeng,
        include_augmented_sigap=args.include_augmented_sigap,
    )

    synthetic_path = args.output_dir / "aksara_sft_synthetic.jsonl"
    synthetic_rows = generate_synthetic_records(
        output_path=synthetic_path,
        model_name=args.synthetic_model,
        target_count=args.synthetic_target,
        seed=args.seed,
    )

    final_sft = deduplicate_records(base_sft + synthetic_rows)
    sigap_rows = [row for row in final_sft if row["category"] == "sigap"]
    dpo_rows = build_dpo_pairs(final_sft, target_count=args.dpo_target, seed=args.seed)
    pretrain_rows = list(iter_pretrain_seed_documents(final_sft))

    written = {
        "sft_base": write_jsonl(args.output_dir / "aksara_sft_base.jsonl", base_sft),
        "sft_synthetic": write_jsonl(synthetic_path, synthetic_rows),
        "sft_final": write_jsonl(args.output_dir / "aksara_sft_final.jsonl", final_sft),
        "sigap": write_jsonl(args.output_dir / "aksara_sigap.jsonl", sigap_rows),
        "dpo": write_jsonl(args.output_dir / "aksara_dpo.jsonl", dpo_rows),
        "pretrain_seed": write_jsonl(args.output_dir / "aksara_pretrain_seed.jsonl", pretrain_rows),
    }

    manifest = {
        "paths": {
            "output_dir": str(args.output_dir),
            "sft_base": str(args.output_dir / "aksara_sft_base.jsonl"),
            "sft_synthetic": str(synthetic_path),
            "sft_final": str(args.output_dir / "aksara_sft_final.jsonl"),
            "sigap": str(args.output_dir / "aksara_sigap.jsonl"),
            "dpo": str(args.output_dir / "aksara_dpo.jsonl"),
            "pretrain_seed": str(args.output_dir / "aksara_pretrain_seed.jsonl"),
        },
        "counts": written,
        "base_stats": base_stats,
        "final_sources": dict(Counter(row["source"] for row in final_sft)),
        "final_categories": dict(Counter(row["category"] for row in final_sft)),
        "synthetic_model": args.synthetic_model if args.synthetic_target else None,
        "synthetic_target": args.synthetic_target,
    }
    (args.output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    LOG.info("Unified dataset build complete")
    for key, value in written.items():
        LOG.info("  %s: %s", key, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
