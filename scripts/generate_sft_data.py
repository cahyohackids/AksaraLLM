#!/usr/bin/env python3
"""
AksaraLLM SFT Generator
Uses GPT-4o-mini to programmatically generate 100K high-quality instruction tuning pairs in Indonesian.
Handles concurrency, rate-limits (via asyncio.Semaphore and exponential backoff), and structured output.

Usage:
    export OPENAI_API_KEY="your-key-here"
    python generate_sft_data.py --target-count 100000 --output-file ../data/sft_100k.jsonl
"""

import os
import sys
import json
import uuid
import time
import asyncio
import argparse
from typing import List, Dict

try:
    from openai import AsyncOpenAI
    import openai
except ImportError:
    print("❌ Error: Please install openai (`pip install openai`)")
    sys.exit(1)


# ─── SEED PROMPTS & CATEGORIES ───
CATEGORIES = [
    "Matematika & Logika",
    "Pemrograman & Coding",
    "Penulisan Kreatif (Puisi, Cerpen)",
    "Pengetahuan Umum Indonesia (Geografi, Sejarah)",
    "Penerjemahan (Daerah/Inggris ke Indonesia)",
    "Tanya Jawab Sehari-hari",
    "Sains & Kedokteran",
    "Agrikultur & Ekonomi Indonesia"
]

SYSTEM_PROMPT = """Anda adalah asisten AI pembuat dataset SFT (Supervised Fine-Tuning) untuk model bahasa Indonesia.
Tugas Anda adalah menghasilkan 5 pasangan instruksi (prompt) dan jawaban (response) yang berkualitas tinggi dalam kategori yang diberikan.
Instruksi harus beragam: berupa pertanyaan, perintah, teka-teki, atau studi kasus.
Jawaban harus akurat, komprehensif, dan menggunakan Bahasa Indonesia yang baik dan benar.

KEMBALIKAN OUTPUT HARUS DALAM FORMAT JSON BERIKUT (TANPA MARKDOWN MENGELILINGI):
{
  "data": [
    {"instruction": "...", "response": "..."},
    ...
  ]
}
"""

async def generate_batch(client: AsyncOpenAI, category: str, semaphore: asyncio.Semaphore, max_retries: int = 5) -> List[Dict]:
    """Generates a batch of instruct-response pairs via API with backoff."""
    prompt = f"Hasilkan 5 pasangan SFT berkualitas tinggi untuk kategori: {category}."
    
    async with semaphore:  # Limit concurrent API calls
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model="gpt-4o-mini",  # Cheapest and fast
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    response_format={"type": "json_object"},
                    timeout=30
                )
                
                content = response.choices[0].message.content
                data = json.loads(content)
                return data.get("data", [])
                
            except openai.RateLimitError:
                delay = 2 ** attempt
                # print(f"  [Rate Limit] Backing off for {delay}s...")
                await asyncio.sleep(delay)
            except Exception as e:
                # print(f"  [Error] {str(e)[:50]}... Retrying.")
                await asyncio.sleep(1)
                
        return []


async def run_generation(target_count: int, output_file: str, concurrency: int):
    """Main loop for generating the full dataset."""
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY environment variable is not set.")
        return

    client = AsyncOpenAI()
    semaphore = asyncio.Semaphore(concurrency)
    
    saved_count = 0
    
    # Check if we should resume from existing file
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            saved_count = sum(1 for _ in f)
        if saved_count > 0:
            print(f"🔄 Resuming from existing file: {saved_count:,} pairs already generated.")
            
    if saved_count >= target_count:
        print(f"✅ Target {target_count:,} already reached in {output_file}.")
        return

    t0 = time.time()
    
    print(f"\n🚀 Starting Massive SFT Generation (Target: {target_count:,})")
    
    with open(output_file, "a", encoding="utf-8") as f:
        # Loop until target is met
        while saved_count < target_count:
            # We fire `concurrency` number of tasks in a batch
            batch_size = min(concurrency, (target_count - saved_count) // 5 + 1)
            # Round-robin selection of categories
            tasks = [
                generate_batch(client, CATEGORIES[i % len(CATEGORIES)], semaphore)
                for i in range(batch_size)
            ]
            
            results = await asyncio.gather(*tasks)
            
            # Save results
            for batch_data in results:
                for pair in batch_data:
                    if saved_count >= target_count:
                        break
                        
                    if "instruction" in pair and "response" in pair:
                        doc = {
                            "id": f"sft_gpt_{uuid.uuid4().hex[:8]}",
                            "instruction": pair["instruction"],
                            "response": pair["response"],
                            "source": "synthetic_gpt4o_mini"
                        }
                        f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                        saved_count += 1
                        
            # Log progress
            elapsed = time.time() - t0
            rate = saved_count / elapsed if elapsed > 0 else 0
            print(f"  [GenSFT] {saved_count:,} / {target_count:,} pairs | {rate:.1f} pairs/sec")
            
    print(f"✅ Finished! {saved_count:,} synthetic pairs saved to {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-count", type=int, default=1000, help="Number of instruction-response pairs to generate")
    parser.add_argument("--output-file", type=str, default="./data/synthetic_sft.jsonl", help="Output JSONL file")
    parser.add_argument("--concurrency", type=int, default=50, help="Maximum concurrent API calls")
    args = parser.parse_args()
    
    out_dir = os.path.dirname(args.output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    asyncio.run(run_generation(args.target_count, args.output_file, args.concurrency))


if __name__ == "__main__":
    main()
