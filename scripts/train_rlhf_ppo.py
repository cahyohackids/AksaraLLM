#!/usr/bin/env python3
"""
AksaraLLM: Reinforcement Learning from Human Feedback (RLHF / PPO)
Uses the `trl` library to align AksaraLLM with human/reward model preferences.

Prerequisites:
  pip install trl peft accelerate xformers

Usage:
  python train_rlhf_ppo.py --model-path ../aksarallm-200m-sft-hf --reward-model-path some_reward_model
"""

import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, set_seed

def build_dataset(dataset_name="Anthropic/hh-rlhf", split="train"):
    """
    Builds the dataset for PPO. 
    By default uses a standard prompt dataset.
    """
    print(f"Loading dataset {dataset_name} for PPO Prompts...")
    ds = load_dataset(dataset_name, split=split)
    
    # We only need the prompt to generate responses -> score -> backprop.
    def extract_prompt(example):
        # Extract everything before the Assistant's last response
        text = example["chosen"]
        prompt_end = text.rfind("\n\nAssistant:")
        if prompt_end != -1:
            return {"query": text[:prompt_end + 12]}
        return {"query": text}
        
    ds = ds.map(extract_prompt)
    ds = ds.rename_column("query", "prompt")
    return ds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="HF format AksaraLLM (SFT stage)")
    parser.add_argument("--reward-model-path", default="OpenAssistant/reward-model-deberta-v3-large-v2", help="HF Reward Model")
    parser.add_argument("--output-dir", default="./aksarallm-rlhf-ppo", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mini-batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1.41e-5)
    args = parser.parse_args()

    set_seed(42)

    print("🚀 Initializing RLHF/PPO Engine...")

    # 1. PPO Configuration
    ppo_config = PPOConfig(
        model_name=args.model_path,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        gradient_accumulation_steps=args.batch_size // args.mini_batch_size,
        optimize_cuda_cache=True,
    )

    # 2. Tokenizer (Our AksaraLLM uses [INST] / [BOS])
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Load Models
    print("📦 Loading Active Policy Model & Reference (Frozen) Model...")
    # TRL automatically loads the reference model as a frozen copy
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    print(f"⚖️  Loading Reward Model: {args.reward_model_path}...")
    rm_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    reward_model.eval()

    # 4. Dataset
    dataset = build_dataset()
    
    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["prompt"])
        return sample
    
    dataset = dataset.map(tokenize, batched=False)
    dataset.set_format(type="torch")

    # 5. PPO Trainer
    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])
        
    trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=collator,
    )

    # Generation kwargs for policy rollouts
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "max_new_tokens": 128,
    }

    # 6. PPO Loop
    print("🌀 Commencing RLHF Proximal Policy Optimization Loop...")
    for epoch, batch in enumerate(trainer.dataloader):
        query_tensors = batch["input_ids"]

        # Step 1: Policy generates response
        response_tensors = trainer.generate(query_tensors, **generation_kwargs)
        batch["response"] = [tokenizer.decode(r.squeeze(), skip_special_tokens=True) for r in response_tensors]

        # Step 2: Compute Rewards
        texts = [q + r for q, r in zip(batch["prompt"], batch["response"])]
        rm_inputs = rm_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(reward_model.device)
        
        with torch.no_grad():
            rewards_p = reward_model(**rm_inputs).logits.squeeze(-1)
        
        # We must align rewards output to list of tensors for TRL
        rewards = [rewards_p[i] for i in range(len(rewards_p))]

        # Step 3: PPO Gradient Update
        stats = trainer.step(query_tensors, response_tensors, rewards)
        
        trainer.log_stats(stats, batch, rewards)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Reward: {torch.mean(rewards_p):.4f} | KL penalty: {stats['objective/kl']:.4f}")
            
        if epoch >= 200: # Demontration limit
            break

    print("💾 Saving PPO aligned model...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("✅ RLHF PPO Phase Complete!")

if __name__ == "__main__":
    main()
