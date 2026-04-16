#!/usr/bin/env python3
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 AksaraLLM DATA QUALITY AUDITOR
 Deep clean 483K → High-quality dataset
 
 Tasks:
 ✅ Remove near-duplicates (fuzzy matching)
 ✅ Score every sample (0-1 quality)
 ✅ Remove paraphrase-heavy junk
 ✅ Detect non-Indonesian / English-only
 ✅ Remove too-short / too-long
 ✅ Rebalance source distribution
 ✅ Upload clean dataset to HuggingFace
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Run:
  pip install datasets huggingface_hub -q
  export HF_TOKEN=hf_xxx
  python3 scripts/data_auditor.py
"""

import re, hashlib, time, os, gc, json
from pathlib import Path
from datetime import datetime
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path(os.environ.get("AKSARALLM_OUTPUT_DIR", PROJECT_ROOT / "outputs"))
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ====================================================================
#  CONFIG
# ====================================================================
HF_TOKEN = os.environ.get("HF_TOKEN")
SFT_REPO = "AksaraLLM/aksara-mega-sft-v5"
CLEAN_REPO = "AksaraLLM/aksara-sft-clean-v1"
DPO_REPO = "AksaraLLM/aksara-dpo-id-v4"
DPO_CLEAN_REPO = "AksaraLLM/aksara-dpo-clean-v1"

# Quality thresholds
MIN_INSTRUCTION_LEN = 8
MIN_OUTPUT_LEN = 30
MAX_OUTPUT_LEN = 3000
MIN_QUALITY = 0.45
MAX_PARAPHRASE_RATIO = 0.5  # Max 50% paraphrase data

INDO_WORDS = {"yang", "dan", "di", "ini", "itu", "dengan", "untuk",
              "dari", "pada", "adalah", "dalam", "tidak", "akan",
              "juga", "sudah", "bisa", "oleh", "ada", "atau",
              "saya", "anda", "kamu", "mereka", "kami", "kita",
              "sangat", "lebih", "karena", "tetapi", "bahwa",
              "seperti", "harus", "banyak", "telah", "dapat",
              "secara", "tersebut", "menjadi", "sebuah", "antara",
              "tentang", "namun", "serta", "beberapa", "setiap"}

BAD_WORDS = re.compile(
    r"\b(judi|slot|togel|casino|porno|bokep|xxx|sex|hack|crack|"
    r"phishing|scam|drugs|narkoba|terorisme|bunuh)\b", re.IGNORECASE
)

# ====================================================================
#  LOGGER
# ====================================================================
def log(msg, level="INFO"):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] [{level}] {msg}", flush=True)

# ====================================================================
#  QUALITY SCORER v2 — More sophisticated
# ====================================================================
def score_quality(instruction, output):
    """Score data quality 0.0 to 1.0"""
    s = 0.0
    inst = str(instruction).strip()
    out = str(output).strip()
    
    # Length checks
    if len(out) < MIN_OUTPUT_LEN: return 0.0
    if len(inst) < MIN_INSTRUCTION_LEN: return 0.0
    if len(out) > MAX_OUTPUT_LEN: s -= 0.1
    
    # Output length score
    if len(out) > 50: s += 0.1
    if len(out) > 100: s += 0.1
    if len(out) > 200: s += 0.05
    
    # Instruction quality
    if len(inst) > 15: s += 0.1
    if "?" in inst or "!" in inst: s += 0.05
    
    # Indonesian detection
    words = set(out.lower().split())
    indo_count = len(words & INDO_WORDS)
    if indo_count >= 3: s += 0.15
    if indo_count >= 5: s += 0.1
    if indo_count >= 8: s += 0.05
    
    # Sentence structure
    sentences = [x.strip() for x in out.split(".") if x.strip()]
    if len(sentences) >= 2: s += 0.1
    if len(sentences) >= 3: s += 0.05
    
    # Uniqueness of sentences
    if len(sentences) > 1:
        unique = len(set(sentences))
        ratio = unique / len(sentences)
        if ratio > 0.7: s += 0.1
        if ratio < 0.3: s -= 0.2  # Too repetitive
    
    # Punctuation (well-formed)
    if "." in out: s += 0.03
    if "," in out: s += 0.02
    
    # Bad content
    if BAD_WORDS.search(inst + " " + out):
        return 0.0
    
    # Repetition penalty
    words_list = out.lower().split()
    if len(words_list) > 10:
        word_freq = Counter(words_list)
        most_common_pct = word_freq.most_common(1)[0][1] / len(words_list) * 100
        if most_common_pct > 30: s -= 0.2  # One word > 30% = repetitive
    
    # Template detection (low-effort paraphrase)
    template_starts = [
        "Jelaskan", "Uraikan", "Ceritakan", "Deskripsikan",
        "Berikan penjelasan", "Tolong jelaskan", "Bisakah kamu",
        "Bagaimana cara memahami", "Sebutkan dan jelaskan",
        "Analisis tentang", "Berikan informasi",
    ]
    is_template = any(inst.startswith(t) for t in template_starts)
    if is_template and len(out) < 100:
        s -= 0.15  # Template with short answer = low quality
    
    return max(min(s, 1.0), 0.0)

# ====================================================================
#  DEDUPLICATOR v2 — Fuzzy + content hash
# ====================================================================
def dedup_advanced(data):
    """Remove duplicates with fuzzy matching."""
    log(f"  🔍 Deduplicating {len(data):,} samples...")
    
    seen_exact = set()  # Exact hash
    seen_fuzzy = set()  # Fuzzy hash (first 50 chars normalized)
    unique = []
    
    for d in data:
        inst = str(d.get("instruction", ""))
        out = str(d.get("output", ""))
        
        # Exact duplicate check
        exact_key = hashlib.md5((inst + "|" + out).encode()).hexdigest()
        if exact_key in seen_exact:
            continue
        seen_exact.add(exact_key)
        
        # Fuzzy duplicate check (ignore prefixes like "Jelaskan", "Uraikan")
        cleaned_inst = re.sub(
            r"^(Jelaskan|Uraikan|Ceritakan|Deskripsikan|Berikan penjelasan tentang|"
            r"Tolong jelaskan|Bisakah kamu menjelaskan|Bagaimana cara memahami|"
            r"Sebutkan dan jelaskan|Analisis tentang|Berikan informasi mengenai)\s+",
            "", inst, flags=re.IGNORECASE
        ).strip().lower()[:60]
        
        fuzzy_key = hashlib.md5((cleaned_inst + "|" + out[:50].lower()).encode()).hexdigest()
        if fuzzy_key in seen_fuzzy:
            continue
        seen_fuzzy.add(fuzzy_key)
        
        unique.append(d)
    
    removed = len(data) - len(unique)
    log(f"  ✅ {len(data):,} → {len(unique):,} (removed {removed:,} dupes, {removed/max(len(data),1)*100:.1f}%)")
    return unique

# ====================================================================
#  SOURCE REBALANCER
# ====================================================================
def rebalance(data, max_paraphrase_ratio=0.5):
    """Limit paraphrase data to prevent overfitting on templates."""
    log(f"  ⚖️ Rebalancing sources...")
    
    sources = Counter(d.get("source", "unknown") for d in data)
    log(f"  Before: {dict(sources.most_common(10))}")
    
    total = len(data)
    max_paraphrase = int(total * max_paraphrase_ratio)
    
    paraphrase = [d for d in data if d.get("source") == "paraphrase"]
    non_paraphrase = [d for d in data if d.get("source") != "paraphrase"]
    
    if len(paraphrase) > max_paraphrase:
        # Keep best paraphrases (longest outputs)
        paraphrase.sort(key=lambda x: len(str(x.get("output", ""))), reverse=True)
        paraphrase = paraphrase[:max_paraphrase]
        log(f"  Trimmed paraphrase: {len(paraphrase):,} (was {sources.get('paraphrase', 0):,})")
    
    result = non_paraphrase + paraphrase
    
    sources_after = Counter(d.get("source", "unknown") for d in result)
    log(f"  After: {dict(sources_after.most_common(10))}")
    
    return result

# ====================================================================
#  DPO QUALITY CHECK
# ====================================================================
def clean_dpo(dpo_data):
    """Clean DPO pairs."""
    log(f"  🎯 Cleaning DPO data ({len(dpo_data):,} pairs)...")
    clean = []
    
    for d in dpo_data:
        prompt = str(d.get("prompt", ""))
        chosen = str(d.get("chosen", ""))
        rejected = str(d.get("rejected", ""))
        
        # Basic checks
        if len(prompt) < 5 or len(chosen) < 20 or len(rejected) < 5:
            continue
        
        # Chosen must be significantly different from rejected
        if chosen == rejected:
            continue
        
        # Chosen should be longer/better
        if len(chosen) < len(rejected) * 0.5:
            continue
        
        # Quality check on chosen
        if score_quality(prompt, chosen) < 0.3:
            continue
        
        clean.append(d)
    
    removed = len(dpo_data) - len(clean)
    log(f"  ✅ DPO: {len(dpo_data):,} → {len(clean):,} (removed {removed:,})")
    return clean

# ====================================================================
#  MAIN AUDITOR
# ====================================================================
def main():
    log("━" * 60)
    log("  🔍 AksaraLLM DATA QUALITY AUDITOR")
    log("━" * 60)
    
    from datasets import load_dataset, Dataset
    from huggingface_hub import login, HfApi
    if HF_TOKEN:
        login(token=HF_TOKEN, add_to_git_credential=False)
    else:
        log("HF_TOKEN not set. Upload akan dilewati, hanya audit lokal yang berjalan.", "WARN")
    api = HfApi(token=HF_TOKEN)
    
    # ── Step 1: Load data ──
    log("\n📥 Step 1: Loading SFT data...")
    ds = load_dataset(SFT_REPO, split="train")
    log(f"  Loaded: {len(ds):,} samples")
    
    data = []
    for item in ds:
        d = dict(item)
        data.append(d)
    del ds; gc.collect()
    
    # ── Step 2: Quality scoring ──
    log("\n📊 Step 2: Quality scoring...")
    scored = []
    rejected = 0
    score_dist = Counter()
    
    for d in data:
        inst = str(d.get("instruction", ""))
        out = str(d.get("output", ""))
        score = score_quality(inst, out)
        
        bucket = int(score * 10) / 10  # 0.0, 0.1, ..., 1.0
        score_dist[bucket] += 1
        
        if score >= MIN_QUALITY:
            d["quality_score"] = round(score, 3)
            scored.append(d)
        else:
            rejected += 1
    
    log(f"  Quality distribution:")
    for bucket in sorted(score_dist.keys()):
        count = score_dist[bucket]
        bar = "█" * int(count / max(score_dist.values()) * 30)
        log(f"    {bucket:.1f}: {count:>8,} {bar}")
    log(f"  Kept: {len(scored):,} | Rejected: {rejected:,} ({rejected/max(len(data),1)*100:.1f}%)")
    
    # ── Step 3: Deduplication ──
    log("\n🧹 Step 3: Advanced deduplication...")
    unique = dedup_advanced(scored)
    
    # ── Step 4: Rebalance ──
    log("\n⚖️ Step 4: Source rebalancing...")
    balanced = rebalance(unique, MAX_PARAPHRASE_RATIO)
    
    # ── Step 5: Final stats ──
    log("\n📊 Step 5: Final statistics...")
    avg_inst_len = sum(len(str(d.get("instruction",""))) for d in balanced) / max(len(balanced), 1)
    avg_out_len = sum(len(str(d.get("output",""))) for d in balanced) / max(len(balanced), 1)
    
    sources = Counter(str(d.get("source", "unknown")) for d in balanced)
    
    log(f"\n{'━'*60}")
    log(f"  📊 AUDIT REPORT")
    log(f"{'━'*60}")
    log(f"  Original:      {len(data):>10,}")
    log(f"  After quality:  {len(scored):>10,}")
    log(f"  After dedup:    {len(unique):>10,}")
    log(f"  After balance:  {len(balanced):>10,}")
    log(f"  Reduction:      {(1-len(balanced)/max(len(data),1))*100:>9.1f}%")
    log(f"  Avg inst len:   {avg_inst_len:>10.0f} chars")
    log(f"  Avg output len: {avg_out_len:>10.0f} chars")
    log(f"{'─'*60}")
    log(f"  Source Distribution:")
    for src, count in sources.most_common(15):
        pct = count / len(balanced) * 100
        log(f"    {str(src)[:30]:<30} {count:>8,} ({pct:.1f}%)")
    log(f"{'━'*60}")
    
    # ── Step 6: Upload clean SFT ──
    log(f"\n📤 Step 6: Uploading clean SFT to {CLEAN_REPO}...")
    clean_ds = Dataset.from_list(balanced)
    if HF_TOKEN:
        try:
            api.create_repo(CLEAN_REPO, exist_ok=True, token=HF_TOKEN)
            clean_ds.push_to_hub(CLEAN_REPO, token=HF_TOKEN)
            log(f"  ✅ Uploaded {len(balanced):,} clean SFT samples!")
        except Exception as e:
            log(f"  ⚠️ SFT upload: {e}", "ERROR")
    else:
        log("  ⚠️ Skip SFT upload: HF_TOKEN not set", "WARN")
    
    # ── Step 7: Clean DPO ──
    log(f"\n🎯 Step 7: Cleaning DPO data...")
    try:
        dpo_ds = load_dataset(DPO_REPO, split="train")
        dpo_data = [dict(item) for item in dpo_ds]
        clean_dpo_data = clean_dpo(dpo_data)
        
        dpo_clean = Dataset.from_list(clean_dpo_data)
        if HF_TOKEN:
            api.create_repo(DPO_CLEAN_REPO, exist_ok=True, token=HF_TOKEN)
            dpo_clean.push_to_hub(DPO_CLEAN_REPO, token=HF_TOKEN)
            log(f"  ✅ Uploaded {len(clean_dpo_data):,} clean DPO pairs!")
        else:
            log("  ⚠️ Skip DPO upload: HF_TOKEN not set", "WARN")
        del dpo_ds, dpo_data; gc.collect()
    except Exception as e:
        log(f"  ⚠️ DPO: {e}", "ERROR")
    
    # ── Done ──
    log(f"\n{'━'*60}")
    log(f"  🏁 AUDIT COMPLETE!")
    log(f"  Clean SFT: {CLEAN_REPO} ({len(balanced):,})")
    log(f"  Clean DPO: {DPO_CLEAN_REPO}")
    log(f"{'━'*60}")

if __name__ == "__main__":
    main()
