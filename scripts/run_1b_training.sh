#!/bin/bash
# Shell script wrapper for launching 1B AksaraLLM Distributed Training
# Utilizes HuggingFace Accelerate with Fully Sharded Data Parallel (FSDP)

set -e

# Configuration 
MODEL_MODE="1b"
DATA_FILE="../data_aksara/oscar_id.jsonl" # Target 10GB dataset
OUTPUT_DIR="../models/aksarallm_1b"

# Training Hyperparams
BATCH_SIZE=4
GRAD_ACCUM=32      # Total effective batch size = 4 * 32 * num_gpus (e.g. 8 GPUs = 1024)
LR="1e-4"
EPOCHS=1

echo "=========================================================="
echo "🚀 Initializing AksaraLLM 1B Parameter Distributed Trainer"
echo "=========================================================="

# 1. Determine environment (Kaggle/Colab vs Cloud TPU vs local Multi-GPU)
# If accelerating directly inside Kaggle, one could use simple accelerate launch.
# Here's a robust command configured to use FSDP automatically mapping resources.

COMMAND="accelerate launch \
    --use_fsdp \
    --fsdp_auto_wrap_policy TRANSFORMER_BASED_WRAP \
    --fsdp_transformer_layer_cls_to_wrap TransformerBlock \
    --fsdp_backward_prefetch BACKWARD_PRE \
    --fsdp_state_dict_type FULL_STATE_DICT \
    --mixed_precision bf16 \
    scripts/train_distributed.py \
    --mode ${MODEL_MODE} \
    --data-file ${DATA_FILE} \
    --output-dir ${OUTPUT_DIR} \
    --batch-size ${BATCH_SIZE} \
    --grad-accum ${GRAD_ACCUM} \
    --learning-rate ${LR} \
    --epochs ${EPOCHS}"

echo "Executing:"
echo $COMMAND

# Execute
$COMMAND
