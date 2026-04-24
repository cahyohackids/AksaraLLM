# AksaraLLM 20B Blueprint

## Recommendation

Build the first 20B AksaraLLM as a dense decoder-only model, not MoE.

Why:

- The current AksaraLLM code already follows the modern dense stack: `RMSNorm + RoPE + SwiGLU + GQA`.
- Qwen2/Qwen2.5 and Mistral Small show that this recipe scales well into the 20B to 30B range.
- Dense is the lowest-risk path if the target is "besar tapi stabil" rather than "paling eksperimental".

Recommended v1 architecture:

- `vocab_size`: `131072`
- `n_embd`: `6144`
- `n_inner`: `20480`
- `n_layers`: `42`
- `n_heads`: `48`
- `n_kv_heads`: `8`
- `head_dim`: `128`
- `norm`: `RMSNorm`
- `activation`: `SwiGLU`
- `position`: `RoPE`
- `attention`: `GQA`
- `tie_embeddings`: `true`
- `estimated_params`: `20.36B`

Parameter count math (verify before any launch):

- Per-layer attention (Q+K+V+O, no bias, GQA 48:8): `88,080,384`
- Per-layer SwiGLU FFN (`3 * d * n_inner`): `377,487,360`
- Per-layer RMSNorms (`2 * d`): `12,288`
- Per-layer total: `465,580,032`
- 42 layers: `19,554,361,344`
- Tied embedding (`V * d`): `805,306,368`
- Final RMSNorm: `6,144`
- **Grand total: `20,359,673,856` (~20.36 B)**

The earlier `n_inner=16384` gave 17.19 B (off from the "20B" target by ~2.6 B). `n_inner=20480` is the smallest TPU-friendly multiple of 256 that reaches ≥20 B with the other dimensions fixed.

This is intentionally close to the design language used by Qwen2/Qwen2.5 and Mistral-class open models, while staying simple enough for the current AksaraLLM codebase.

## Context Strategy

Do not train 128K from step 1.

Recommended sequence curriculum:

1. Stage 1 pretrain at `4096` to `8192` context.
2. Stage 2 extend to `32768` near the end of pretraining.
3. Use `YaRN` only after the base model is already healthy, mainly for deployment or a late long-context phase.

Reason:

- Qwen2 reports expanding context from `4096` to `32768` during the concluding phase of pretraining, not from the beginning.
- Qwen2.5 model cards expose `131072` context through `YaRN`, with `32768` as the original max position size.

For AksaraLLM, the safe default is:

- `train_context_main`: `8192`
- `train_context_late`: `32768`
- `inference_context_target`: `131072`
- `rope_theta`: `1000000`

## Token Budget

Use three planning numbers, not one:

- Minimum viable: `400B` tokens
- Recommended serious run: `600B` tokens
- Strong run: `1T` tokens

Why these numbers:

- Chinchilla scaling says compute-optimal training scales tokens roughly with parameters. For a `20B` model, that implies about `400B` tokens.
- Qwen2 reports dense models up to `72B` trained on about `7T` tokens.
- Llama 3 reports `15T+` pretraining tokens for both `8B` and `70B`.
- Qwen2.5 says its family is pretrained on datasets of up to `18T` tokens.

Interpretation:

- `400B` is the "do not go below this if training from scratch" floor.
- `600B` is the practical target for a first strong Indonesian-first 20B run.
- `1T` is where the model is much less likely to feel undertrained.

## Data Mix

For a `600B` token run, use this target mixture:

- `35-45%` high-quality global web
- `20-25%` multilingual web
- `12-18%` Indonesia + SEA targeted text
- `10-15%` code
- `10-15%` books, papers, wiki, and reference text

Important:

- Keep Indonesia + SEA as a dedicated bucket, not only a side effect of multilingual sampling.
- Do not use your current `AksaraLLM/aksara-unified-datasets-v1` as the main 20B pretraining corpus. That dataset is for SFT, DPO, SIGAP, and pretrain seeding, not web-scale base pretraining.

Practical Indonesia-first interpretation for `600B` tokens:

- Dedicated Indonesian + local-language slice should land around `70B` to `100B` tokens.
- The rest of Indonesian exposure can still come indirectly from multilingual sources such as FineWeb2 and CulturaX.

## Source Priority

Primary sources to prioritize:

1. `HuggingFaceFW/fineweb`
   - Best candidate for the high-quality general web bucket.
   - English-heavy, large, deduplicated, reproducible.
2. `HuggingFaceFW/fineweb-2`
   - Best candidate for multilingual scale.
   - Includes a very large Indonesian subset.
3. `uonlp/CulturaX`
   - Strong multilingual supplement, especially when you want broader non-English coverage.
4. `SEACrowd/indo4b`
   - Strong Indonesian-focused booster set.
5. `SEACrowd/cc100`
   - Useful for Javanese, Sundanese, Malay, Thai, Vietnamese, Khmer, Lao, Tagalog, etc.
6. `bigcode/the-stack-v2`
   - Best open code-scale source, but license handling must be respected.
7. `allenai/dolma`
   - Great reference mixture for books, papers, wiki, and balanced non-web text.
8. `togethercomputer/RedPajama-Data-V2`
   - Optional backup or diversity source, not the first choice if FineWeb is available.

## Concrete 20B Data Recipe

If you want a practical first build order:

1. Build the base corpus from:
   - FineWeb
   - FineWeb2
   - CulturaX
   - Dolma-style reference subsets
2. Add dedicated Indonesia/SEA oversampling from:
   - FineWeb2 Indonesian
   - Indo4B
   - SEACrowd CC100 subsets for `ind`, `jav`, `sun`, `zlm`, `vie`, `tha`, `khm`, `lao`, `tgl`
   - clean local corpora you own
3. Add code as a separate bucket from:
   - The Stack v2
   - permissive-license filtering only if you want lower compliance risk
4. Keep post-training data separate:
   - `aksara_sft_final.jsonl`
   - `aksara_dpo.jsonl`
   - `aksara_sigap.jsonl`

## What To Avoid

- Do not keep `vocab_size=32000` for a multilingual 20B model.
- Do not train long-context from the first step.
- Do not mix raw noisy crawl data without aggressive dedupe and filtering.
- Do not let SFT-style conversational data dominate base pretraining.
- Do not jump to MoE first if the real goal is a stable 20B AksaraLLM.

## Suggested Next Moves

1. Freeze the tokenizer target at `128K` to `152K`.
2. Build a manifest-driven pretraining corpus with domain buckets and token quotas.
3. Start with a `600B` token plan.
4. Reserve the existing Aksara unified dataset for SFT/DPO after base pretraining or late continued pretraining.

## Sources

- Chinchilla scaling law: https://arxiv.org/abs/2203.15556
- Qwen2 technical report: https://arxiv.org/abs/2407.10671
- Qwen2.5 release blog: https://qwenlm.github.io/blog/qwen2.5/
- Qwen2.5-32B model card: https://huggingface.co/Qwen/Qwen2.5-32B-Instruct
- Llama 3 70B model card: https://huggingface.co/meta-llama/Meta-Llama-3-70B
- FineWeb dataset card: https://huggingface.co/datasets/HuggingFaceFW/fineweb
- FineWeb2 dataset card: https://huggingface.co/datasets/HuggingFaceFW/fineweb-2
- Dolma dataset card: https://huggingface.co/datasets/allenai/dolma
- CulturaX dataset card: https://huggingface.co/datasets/uonlp/CulturaX
- Indo4B dataset card: https://huggingface.co/datasets/SEACrowd/indo4b
- SEACrowd CC100 dataset card: https://huggingface.co/datasets/SEACrowd/cc100
- The Stack v2 dataset card: https://huggingface.co/datasets/bigcode/the-stack-v2
- RedPajama Data V2 dataset card: https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2
