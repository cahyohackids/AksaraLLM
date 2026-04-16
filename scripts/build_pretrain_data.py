#!/usr/bin/env python3
"""
AksaraLLM Data Pipeline: 10x Scale
Downloads OSCAR Indonesian subset, applies heuristic filtering,
and includes an async scaffolding for web scraping.

Usage:
    python build_pretrain_data.py --target-gb 10 --output-dir ../data
"""

import os
import sys
import json
import uuid
import time
import asyncio
import argparse
from typing import List, Dict, Set

# For fast text processing
import re

# Fallback basic Indonesian checker
INDO_STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk", "pada", "adalah", 
    "dengan", "tidak", "akan", "bisa", "sebagai", "dalam", "bahwa", "tersebut", 
    "oleh", "saat", "juga", "telah", "lebih", "kami", "kita"
}

def is_indonesian(text: str) -> bool:
    """Fast heuristic validation for Indonesian text."""
    words = text.lower().split()
    if not words:
        return False
    
    stop_count = sum(1 for w in words if w in INDO_STOPWORDS)
    ratio = stop_count / len(words)
    # If at least 5% of the words are common Indonesian stopwords, pass.
    return ratio > 0.05

def clean_text(text: str) -> str:
    """Basic text standardization."""
    text = re.sub(r'\s+', ' ', text)
    # Remove extremely long words (often base64 or garbage)
    words = [w for w in text.split() if len(w) < 50]
    return " ".join(words).strip()


# ─── 1. OSCAR DOWNLOADER ───
def process_oscar(target_bytes: int, output_file: str):
    """Streams OSCAR dataset from HuggingFace to disk."""
    print(f"\n📦 Connecting to HuggingFace Datasets (OSCAR-2301 id)...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ Error: Please install datasets (`pip install datasets`)")
        sys.exit(1)

    ds = load_dataset("oscar-corpus/OSCAR-2301", "id", split="train", streaming=True)
    
    saved_bytes = 0
    saved_docs = 0
    skipped = 0
    
    import hashlib
    # Python's built-in hash() is randomized per process. 
    # We MUST use hashlib for stable, deterministic deduplication.
    seen_hashes: Set[str] = set()

    with open(output_file, "w", encoding="utf-8") as f:
        t0 = time.time()
        for i, item in enumerate(ds):
            raw_text = item.get("text", "")
            
            # 1. Clean
            text = clean_text(raw_text)
            
            # 2. Length Filter
            if len(text) < 200:
                skipped += 1
                continue
                
            # 3. Lang Filter
            if not is_indonesian(text):
                skipped += 1
                continue
                
            # 4. Exact Deduplication via md5 hash
            thash = hashlib.md5(text.encode('utf-8')).hexdigest()
            if thash in seen_hashes:
                skipped += 1
                continue
            seen_hashes.add(thash)
            
            # 5. Save
            doc = {
                "id": f"oscar_{uuid.uuid4().hex[:8]}",
                "text": text,
                "source": "oscar-2301"
            }
            line = json.dumps(doc, ensure_ascii=False) + "\n"
            f.write(line)
            
            saved_bytes += len(line.encode("utf-8"))
            saved_docs += 1
            
            # Log progress
            if saved_docs % 1000 == 0:
                elapsed = time.time() - t0
                gb_saved = saved_bytes / (1024**3)
                gb_target = target_bytes / (1024**3)
                rate = (saved_bytes / 1024**2) / elapsed
                print(f"  [OSCAR] {saved_docs:,} docs | {gb_saved:.2f}/{gb_target:.2f} GB | {rate:.1f} MB/s | Skipped: {skipped}")
            
            if saved_bytes >= target_bytes:
                break
                
    print(f"✅ OSCAR Target reached: {saved_docs:,} documents saved to {output_file}")


# ─── 2. ASYNC WEB SCRAPER ───
async def fetch_article(session, url: str) -> str:
    """Fetch and extract article text using BeautifulSoup."""
    try:
        async with session.get(url, timeout=10) as response:
            if response.status != 200:
                return ""
            html = await response.text()
            
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            
            # Simple heuristic: grab all paragraphs
            paragraphs = soup.find_all("p")
            text = "\n".join([p.get_text() for p in paragraphs])
            return text
    except Exception:
        return ""

async def run_scraper(urls: List[str], output_file: str):
    """Run async scraper on target URLs."""
    try:
        import aiohttp
        from bs4 import BeautifulSoup
    except ImportError:
        print("❌ Error: Scraper needs aiohttp and beautifulsoup4 (`pip install aiohttp beautifulsoup4`)")
        return

    print(f"\n🌐 Starting targeted Async Web Scraper...")
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_article(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
        
        saved = 0
        with open(output_file, "w", encoding="utf-8") as f:
            for text in results:
                text = clean_text(text)
                if len(text) > 300 and is_indonesian(text):
                    doc = {
                        "id": f"scrape_{uuid.uuid4().hex[:8]}",
                        "text": text,
                        "source": "web_scraper"
                    }
                    f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    saved += 1
        print(f"✅ Scraping done. {saved}/{len(urls)} valid articles saved to {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-gb", type=float, default=1.0, help="Target data size in Gigabytes")
    parser.add_argument("--output-dir", type=str, default="./data", help="Output directory")
    parser.add_argument("--scrape-only", action="store_true", help="Only run the scraper template")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    target_bytes = int(args.target_gb * 1024**3)
    
    if not args.scrape_only:
        oscar_file = os.path.join(args.output_dir, "oscar_id.jsonl")
        process_oscar(target_bytes, oscar_file)
    
    # Scaffold for scraping targeted sites
    scraper_file = os.path.join(args.output_dir, "scraped_data.jsonl")
    example_urls = [
        "https://id.wikipedia.org/wiki/Kecerdasan_buatan",
        "https://id.wikipedia.org/wiki/Indonesia",
        "https://id.wikipedia.org/wiki/Bahasa_Indonesia"
    ]
    asyncio.run(run_scraper(example_urls, scraper_file))

if __name__ == "__main__":
    main()
