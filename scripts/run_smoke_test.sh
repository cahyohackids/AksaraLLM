#!/bin/bash
# Run a 20-step smoke test of the 20B pretrain runner.
# Designed to fit on v6e-8 (8 chips, 256 GB HBM) using a TINY proxy model
# that reuses the real data + tokenizer path but substitutes a ~200M model
# config for speed. Validates:
#   - tokenizer loads and produces 131K vocab
#   - parquet glob iterates
#   - EasyDeL trainer JITs and takes 20 gradient steps
#   - Orbax saves & resumes a sharded checkpoint
#   - W&B receives metrics
#
# Usage on a TPU VM (pre-reqs: pip install easydel jax[tpu] orbax-checkpoint wandb):
#   bash scripts/run_smoke_test.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${TOKENIZER:=Ezekiel999/aksara-tokenizer-20b}"
: "${CORPUS_GLOB:=${REPO_ROOT}/outputs/pretrain/smoke/*/*.parquet}"
: "${OUTPUT_DIR:=${REPO_ROOT}/outputs/smoke-ckpt}"
: "${WANDB_PROJECT:=aksarallm-smoke}"

# Tiny proxy config: same tokenizer vocab, but 2 layers × dim 256. Keeps the
# runner + sharding + checkpoint code on its real code paths without eating
# a full 20B of HBM.
TMP_CONFIG="$(mktemp --suffix=.json)"
cat > "${TMP_CONFIG}" <<'JSON'
{
  "name": "aksarallm-smoke",
  "family": "decoder-only-transformer",
  "estimated_params": 0,
  "architecture": {
    "vocab_size": 131072,
    "n_embd": 256,
    "n_inner": 512,
    "n_layers": 2,
    "n_heads": 4,
    "n_kv_heads": 2,
    "head_dim": 64,
    "norm_type": "rmsnorm",
    "activation": "swiglu",
    "positional_encoding": "rope",
    "attention_type": "gqa",
    "attention_bias": false,
    "tie_embeddings": true,
    "rope_theta": 1000000
  },
  "sequence_plan": {
    "train_context_main": 1024
  }
}
JSON

echo "[smoke] config: ${TMP_CONFIG}"
echo "[smoke] tokenizer: ${TOKENIZER}"
echo "[smoke] corpus_glob: ${CORPUS_GLOB}"
echo "[smoke] output_dir: ${OUTPUT_DIR}"

python3 "${REPO_ROOT}/scripts/train_20b_pretrain.py" \
    --config "${TMP_CONFIG}" \
    --tokenizer "${TOKENIZER}" \
    --corpus-glob "${CORPUS_GLOB}" \
    --output-dir "${OUTPUT_DIR}" \
    --seq-len 1024 \
    --global-batch-tokens 131072 \
    --tp-size 2 \
    --optimizer adamw \
    --peak-lr 1e-4 \
    --warmup-steps 10 \
    --max-steps 20 \
    --ckpt-every 10 \
    --ckpt-keep 2 \
    --ckpt-permanent-every 1000 \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-run-name "smoke-$(date +%Y%m%d-%H%M%S)" \
    --smoke-test

rm -f "${TMP_CONFIG}"
echo "[smoke] done."
