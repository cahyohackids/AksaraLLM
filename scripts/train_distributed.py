#!/usr/bin/env python3
"""
AksaraLLM Distributed Training Pipeline (FSDP / DDP / TPU)
Powered by HuggingFace Accelerate for seamless Multi-GPU and TPU scaling.

Features:
- Fully Sharded Data Parallel (FSDP) to train 1B+ models on limited VRAM
- Gradient Accumulation for massive simulated batch sizes
- Mixed Precision (bfloat16)
- Gradient Checkpointing integration

Usage:
    accelerate launch scripts/train_distributed.py --mode 1b --batch-size 8 --learning-rate 3e-4
"""

import os
import sys
import math
import json
import argparse
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator
from accelerate.utils import set_seed
from datasets import load_dataset

# Ensure src/ is in path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
from aksarallm import AksaraLLM, ModelConfig, get_config, AksaraTokenizer

def get_lr(it, warmup_iters, max_iters, learning_rate, min_lr):
    """Cosine learning rate schedule with warmup."""
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="200m", help="Model size config")
    parser.add_argument("--data-file", type=str, required=True, help="Path to training data (jsonl)")
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ckpt-steps", type=int, default=1000)
    args = parser.parse_args()

    # Initialize Accelerator
    # Handled via `accelerate config` externally
    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision="bf16",
    )
    set_seed(args.seed)

    accelerator.print(f"🚀 Starting AksaraLLM Distributed Training: {args.mode.upper()}")

    # Setup Tokenizer & Model
    tokenizer = AksaraTokenizer()
    config = get_config(args.mode)
    model = AksaraLLM(config)
    
    # ⚡ Enable explicit gradient checkpointing to smash VRAM usage!
    model.enable_gradient_checkpointing()
    
    accelerator.print(f"📦 Model Parameters: {model.num_trainable_parameters / 1e6:.1f} M")

    # Dataset & Dataloader
    dataset = load_dataset("json", data_files=args.data_file, split="train")
    
    def tokenize_function(example):
        text = example.get("text", example.get("response", ""))
        ids = tokenizer.encode(text, add_bos=True, add_eos=True, max_length=config.max_seq_len + 1)
        
        # Create shifting for Next Token Prediction
        x = ids[:-1]
        y = ids[1:]
        
        # Padding
        pad_len = config.max_seq_len - len(x)
        if pad_len > 0:
            x.extend([tokenizer.pad_id] * pad_len)
            y.extend([-100] * pad_len) # -100 is ignored sequence by CrossEntropyLoss
            
        return {"x": x, "y": y}
        
    with accelerator.main_process_first():
        tokenized_dataset = dataset.map(
            tokenize_function, 
            remove_columns=dataset.column_names, 
            num_proc=4,
            desc="Tokenizing dataset"
        )
        tokenized_dataset.set_format("torch")
        
    dataloader = DataLoader(tokenized_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1, betas=(0.9, 0.95))

    # Prepare for Distributed Training (Magic happens here)
    model, optimizer, dataloader = accelerator.prepare(
        model, optimizer, dataloader
    )

    max_steps = (len(dataloader) // args.grad_accum) * args.epochs
    min_lr = args.learning_rate / 10.0
    
    global_step = 0
    os.makedirs(args.output_dir, exist_ok=True)

    # Training Loop
    progress_bar = tqdm(total=max_steps, disable=not accelerator.is_local_main_process)
    
    for epoch in range(args.epochs):
        model.train()
        for step, batch in enumerate(dataloader):
            x = batch["x"]
            y = batch["y"]
            
            # Dynamic LR adjustment
            lr = get_lr(global_step, args.warmup_steps, max_steps, args.learning_rate, min_lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Forward -> Backward with Accumulation
            with accelerator.accumulate(model):
                logits, loss, _ = model(x, targets=y)
                accelerator.backward(loss)
                
                # Gradient Clipping is essential for massive models
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                
                if global_step % 10 == 0:
                    accelerator.print(f"Step {global_step} | LR {lr:.2e} | Loss {loss.item():.4f}")

                # Save Checkpoint
                if global_step % args.ckpt_steps == 0:
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    ckpt_path = os.path.join(args.output_dir, f"step_{global_step}.pt")
                    if accelerator.is_main_process:
                        torch.save({
                            "step": global_step,
                            "state_dict": unwrapped_model.state_dict(),
                        }, ckpt_path)
                    accelerator.print(f"💾 Checkpoint saved: {ckpt_path}")

    # Final Save
    accelerator.wait_for_everyone()
    final_path = os.path.join(args.output_dir, "final.pt")
    unwrapped_model = accelerator.unwrap_model(model)
    if accelerator.is_main_process:
        torch.save(unwrapped_model.state_dict(), final_path)
    accelerator.print("✅ Training Complete!")

if __name__ == "__main__":
    main()
