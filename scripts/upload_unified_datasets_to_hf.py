#!/usr/bin/env python3
"""
Upload unified AksaraLLM dataset artifacts to a Hugging Face dataset repo.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi


def build_readme(repo_id: str, folder: Path) -> str:
    manifest_path = folder / "dataset_manifest.json"
    counts: dict[str, int] = {}
    final_sources: dict[str, int] = {}
    final_categories: dict[str, int] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        counts = manifest.get("counts", {})
        final_sources = manifest.get("final_sources", {})
        final_categories = manifest.get("final_categories", {})

    return f"""---
license: apache-2.0
language:
- id
task_categories:
- text-generation
- text-classification
pretty_name: AksaraLLM Unified Datasets v1
---

# {repo_id}

Unified AksaraLLM datasets generated from TPU output.

## Files

- `aksara_sft_base.jsonl`
- `aksara_sft_synthetic.jsonl`
- `aksara_sft_final.jsonl`
- `aksara_sigap.jsonl`
- `aksara_dpo.jsonl`
- `aksara_pretrain_seed.jsonl`
- `dataset_manifest.json`

## Counts

```json
{json.dumps(counts, ensure_ascii=False, indent=2)}
```

## Final Sources

```json
{json.dumps(final_sources, ensure_ascii=False, indent=2)}
```

## Final Categories

```json
{json.dumps(final_categories, ensure_ascii=False, indent=2)}
```
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload unified dataset artifacts to Hugging Face.")
    parser.add_argument("--repo-id", required=True, help="Target dataset repo id, e.g. AksaraLLM/aksara-unified-datasets-v1")
    parser.add_argument("--folder", required=True, type=Path, help="Folder containing generated dataset files")
    parser.add_argument("--private", action="store_true", help="Create repo as private")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set")

    folder = args.folder.expanduser().resolve()
    if not folder.exists():
        raise SystemExit(f"Folder not found: {folder}")

    readme_path = folder / "README.md"
    readme_path.write_text(build_readme(args.repo_id, folder), encoding="utf-8")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )
    api.upload_folder(
        folder_path=str(folder),
        repo_id=args.repo_id,
        repo_type="dataset",
        allow_patterns=["*.jsonl", "*.json", "README.md"],
        commit_message="Upload unified TPU-generated AksaraLLM datasets",
    )
    print(f"https://huggingface.co/datasets/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
