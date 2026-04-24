# AksaraLLM 20B — Multi-Week Execution Plan (Batch C)

> **Audience.** The TPU operator who actually runs the pre-training. This is
> the step-by-step procedure a human follows on their own GCP project after
> the Batch A code PR lands and the Batch B tokenizer is uploaded. Devin
> cannot execute these steps — they require a reserved TPU pod, custody of
> the GCS corpus bucket, and multi-week wall-clock.
>
> **Prerequisites.** Batch A PR merged. Tokenizer uploaded to
> `Ezekiel999/aksara-tokenizer-20b` (Batch B). At least one of:
> v5p-64 (minimum), v5p-128 (recommended), v5p-256 (fastest). TRC free
> quota is fine provided capacity is available; otherwise a paid reservation.
> GCS bucket for corpus + checkpoints in the same region as the TPU pod.
> W&B workspace for metrics.

## TL;DR timeline

Assuming Batch B finishes on day 0:

| Week | Deliverable | Gate |
|------|-------------|------|
| 1 | Corpus v2 built on GCS; manifest shows ≥ 500 B tokens | `preflight_20b.py --min-tokens 500000000000` PASS |
| 1 | 1B smoke on v6e-8; Orbax resume verified | Loss → < 6.0 @ 2 k steps, resume ± 0.01 |
| 2 | Real pretrain phase 1 starts on v5p-64/128 | `jax.device_count()` matches; `run_20b_pretrain.py` at step 0 |
| 2–6 | Pretrain phase 1: 8 k context, ≈ 400 B tokens | Val perplexity monotone ↓; MFU ≥ 35 % on v5p |
| 6 | Context extension phase: 8 k → 32 k via YaRN | Held-out long-context perplexity within 10 % of 8 k |
| 7 | SFT on AksaraLLM-SFT-20B | Eval suite pass-rate ≥ Qwen2-7B-Instruct baseline |
| 8 | DPO/ORPO alignment | Reward-margin > 0; WinRate > 55 % vs. SFT |
| 9 | GGUF export + HF release | `AksaraLLM/AksaraLLM-20B-Instruct` public, model card filled |

Total: **≈ 8–9 weeks on v5p-128**. Double this on v5p-64. **Do not attempt
phase 1 pretrain on v6e-8** — it works mathematically but is ~6× slower per
FLOP than v5p per dollar, and the math says 6–9 months wall-clock for 400 B
tokens. Use v6e-8 for Batch B + smoke tests + SFT/DPO only.

---

## Week 1, Day 1–3: Build the corpus on GCS

The `build_pretrain_corpus_v2.py` pipeline is designed to be launched on a
single n2-highcpu-96 or a small TPU VM (CPU-bound). Target the same GCS
region as the TPU pod to keep egress free.

```bash
# on an n2-highcpu-96 with 3 TB scratch disk
cd ~/AksaraLLM
python scripts/build_pretrain_corpus_v2.py download-assets \
    --assets-dir ~/lid_models

python scripts/build_pretrain_corpus_v2.py build \
    --config configs/aksara_20b_dense.json \
    --output-dir gs://aksarallm-corpus/pretrain \
    --tokenizer Ezekiel999/aksara-tokenizer-20b \
    --lid-model ~/lid_models/lid.176.bin \
    --total-token-budget 600000000000 \
    --num-workers 64 \
    --enable-minhash --minhash-threshold 0.8 \
    --enable-decontamination
```

**Gates (block launch if any fail):**
- `manifest.json`'s `per_source_stats[*].kept_tokens` sum ≥ 500 B.
- Per-bucket realised ratio within ±5 % of target (`data_mix`).
- Fertility (measured on 1 M-word held-out id/en/code) ≤ 1.6 / 1.4 / 2.0.
- Benchmark decontamination log reports non-zero drops (sanity check).

Expected total cost: 4–7 days wall-clock on the highcpu VM, ≈ $400 compute +
egress-free GCS writes.

## Week 1, Day 4–5: 1B smoke-test end-to-end

On **v6e-8** (not the 20B target — a proxy for the code path):

```bash
# provision a v6e-8 via `gcloud compute tpus tpu-vm create`
gcloud compute tpus tpu-vm ssh aksara-smoke-v6e8 --worker=all --command="
  git clone https://github.com/cahyohackids/AksaraLLM && cd AksaraLLM
  pip install -r requirements-tpu.txt easydel orbax-checkpoint wandb datasets
  bash scripts/run_smoke_test.sh
"
```

Smoke test must:
1. Complete 20 steps without OOM or NaN.
2. Write an Orbax checkpoint to the output dir.
3. Stop, restart with `--resume`, observe loss within ±0.01 of pre-stop.
4. Log 20 steps to W&B with `grad_norm`, `tokens_per_sec`, `step_ms`.

If any of the four fails, **do not proceed** — debug on v6e-8 first.
The 20B runner and the smoke test share the same code path, so a failure
at 200 M reliably predicts a failure at 20 B.

## Week 2: Provision the pretrain TPU pod

Preferred: **v5p-128** (`--accelerator-type v5p-128 --zone us-east5-a`). If
TRC only has v5p-64, cut the target token budget to 400 B and double the
wall-clock budget. v5p-256 works too and roughly halves wall-clock.

```bash
gcloud compute tpus tpu-vm create aksara-20b-pretrain \
    --zone us-east5-a --accelerator-type v5p-128 \
    --version v2-alpha-tpuv5 --project aksarallm-tpu

gcloud compute tpus tpu-vm ssh aksara-20b-pretrain --worker=all --command="
  git clone https://github.com/cahyohackids/AksaraLLM && cd AksaraLLM
  pip install -r requirements-tpu.txt easydel orbax-checkpoint wandb datasets gcsfs
  huggingface-cli login --token \$HF_TOKEN
  wandb login \$WANDB_API_KEY
  python scripts/preflight_20b.py \
      --config configs/aksara_20b_dense.json \
      --tokenizer Ezekiel999/aksara-tokenizer-20b \
      --corpus-glob 'gs://aksarallm-corpus/pretrain/*/*.parquet' \
      --output-dir gs://aksarallm-checkpoints/20b \
      --expected-chips 128 --tp-size 4 \
      --min-tokens 400000000000
"
```

`preflight_20b.py` must exit 0 before the next step. If any gate fails,
fix it — not the gate. Each failure represents a real runtime blocker.

## Weeks 2–6: Pretrain phase 1 (8 k context, ~400 B tokens)

```bash
gcloud compute tpus tpu-vm ssh aksara-20b-pretrain --worker=all --command="
  cd AksaraLLM && python scripts/train_20b_pretrain.py \
      --config configs/aksara_20b_dense.json \
      --tokenizer Ezekiel999/aksara-tokenizer-20b \
      --corpus-glob 'gs://aksarallm-corpus/pretrain/*/*.parquet' \
      --output-dir gs://aksarallm-checkpoints/20b \
      --seq-len 8192 \
      --global-batch-tokens 2097152 \
      --tp-size 4 \
      --optimizer adamw \
      --peak-lr 1.5e-4 --warmup-steps 5000 \
      --max-steps 200000 \
      --ckpt-every 500 --ckpt-keep 3 --ckpt-permanent-every 10000 \
      --wandb-project aksarallm-20b --wandb-run-name 'phase1-8k'
"
```

Operational discipline:

- **Restart policy.** TPU preemptions are expected. Orbax async restore
  kicks in on restart; resume is automatic.
- **Health dashboard.** Watch `grad_norm`, `loss_ewm`, `tokens_per_sec`,
  `step_ms` in W&B. Alert if `grad_norm > 5` for > 20 steps in a row, or
  `tokens_per_sec` drops by > 30 % from the warm EMA.
- **Divergence recovery.** If the run diverges (NaN), restart from the
  last permanent checkpoint and drop peak LR by 30 %.
- **Cost guard.** v5p-128 at $x/hour * ~4 weeks. Check daily spend against
  budget; pause if over.

Stop criteria: `loss_ewm < 1.95` *and* `tokens_consumed ≥ 400 B` *and*
val-perplexity monotone-decreasing last 24 h.

## Week 6: Context extension phase (8 k → 32 k via YaRN)

Re-run the same script with `--seq-len 32768 --max-steps 6000
--peak-lr 3e-5 --resume`. Reduce `--global-batch-tokens` to 1048576 so
each microbatch still fits. YaRN scaling is already in the JSON config;
EasyDeL reads it.

Gate: long-context perplexity within 10 % of 8 k perplexity at the same
step, on a held-out 32 k-book set (e.g. wiki book passages).

## Week 7: Supervised fine-tuning (SFT)

SFT data lives at `AksaraLLM/aksara-mega-sft-v5` (existing) + the new
`Ezekiel999/aksara-sft-20b`. The existing `scripts/train_sft_dpo.py`
targets Qwen2.5-1.5B; for the 20B model, use a 20B-shaped EasyDeL SFT
script — see the "TODO: write train_20b_sft.py" note in the PR body.
Until that script exists, use the model architecture saved from
pretrain phase 2 and run SFT with:

- Effective batch = 512 conversations, 4096 context.
- Peak LR = 1e-5 with 200-step warmup then cosine to 1e-6.
- 3 epochs over ~100 k high-quality conversations.

Gate: IndoMMLU ≥ Qwen2-7B-Instruct baseline; CulturaY-eval ≥ 60 %.

## Week 8: Alignment (ORPO or DPO)

**Do not use `copy.deepcopy(model)` for the 20B reference.** The fix that
landed for the 1.5B script (disk-snapshot + bf16 reload) still doesn't
scale to 20B because it keeps two 40 GB copies in HBM. Either:

- **Recommended for 20B:** **ORPO** (Odds-Ratio Preference Optimization) —
  no reference model required, single forward pass. Half the HBM of DPO.
- **Acceptable fallback:** DPO with the reference stored as a *sharded
  param-only* snapshot on a dedicated sub-mesh (model-parallel only, no
  optimizer state). Budget 1.5× the HBM of SFT.
- **Forbidden:** `copy.deepcopy`. At 20B this OOMs the pod.

Gate: reward-margin > 0 on held-out preference set; GPT-4-as-judge
WinRate > 55 % vs. pre-DPO SFT checkpoint.

## Week 9: Export + release

1. Convert EasyDeL/Flax checkpoint → HF Transformers LlamaForCausalLM
   weights (EasyDeL ships `to_torch.py`).
2. Upload to `AksaraLLM/AksaraLLM-20B-Base` and `-Instruct`.
3. Convert to GGUF (Q4_K_M, Q5_K_M, Q8_0) via existing
   `scripts/export_gguf.py`; upload `-GGUF` sibling repo.
4. Fill in the HF model card: intended use, training data sources,
   evaluation numbers, known limitations, license (Apache-2.0).
5. Publish launch blog with W&B run links, corpus manifest, and
   reproducibility checklist.

## Kill-switch criteria

At any point, **abort the pretrain** (don't soldier on) if:
- Loss diverges twice from the same checkpoint after LR drop.
- Estimated total tokens < 300 B at week 5.
- W&B/Orbax stop writing for > 6 h (silent failure).
- TPU pod is preempted and capacity does not return within 72 h.

Aborting at week 5 costs ≈ 60 % of the bill but preserves the checkpoint
for future continuation. Pushing through a diverging run wastes 100 % of
the bill and leaves no usable checkpoint.

## Roles & on-call

- **Operator** (Cahyo): launches, monitors W&B, responds to preemption.
- **Devin** (Batch A+B only): code fixes, data pipeline builds, tokenizer.
  Cannot monitor week-long runs; hand off to operator after smoke test.
- **Reviewer** (volunteer ML eng): spot-checks val perplexity at weeks
  2, 4, 6, and approves the transition from pretrain → SFT.

---

*Last updated alongside the Batch A code PR. If you tweak the runner,
tweak this doc.*
