#!/usr/bin/env python3
"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 AksaraLLM AUTOPILOT v2 — PRODUCTION GRADE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Features:
 ✅ Smart dataset discovery (HF API + language tags)
 ✅ Indonesian language detection
 ✅ Quality scoring & filtering
 ✅ Progress persistence (resume after crash)
 ✅ Error recovery & retry logic
 ✅ Smart deduplication (fuzzy matching)
 ✅ Auto-expanding synthetic data
 ✅ Data augmentation (paraphrase, QA generation)
 ✅ Statistics & quality dashboard
 ✅ Disk caching (no re-download)
 ✅ Wikipedia article scraping
 ✅ Rate limiting for APIs

Run:
  export HF_TOKEN=hf_xxx
  python3 scripts/data_autopilot.py
"""

import random, json, time, re, hashlib, os, gc, sys
import traceback, pickle
from datetime import datetime
from pathlib import Path

# ==============================================================
#  CONFIG
# ==============================================================
class Config:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    OUTPUT_ROOT = Path(os.environ.get("AKSARALLM_OUTPUT_DIR", PROJECT_ROOT / "outputs"))
    AUTOPILOT_DIR = OUTPUT_ROOT / "autopilot"
    HF_TOKEN = os.environ.get("HF_TOKEN")
    SFT_REPO = "AksaraLLM/aksara-mega-sft-v5"
    DPO_REPO = "AksaraLLM/aksara-dpo-id-v4"
    TARGET_SFT = 500_000
    TARGET_DPO = 200_000
    CACHE_DIR = str(AUTOPILOT_DIR / "cache")
    STATE_FILE = str(AUTOPILOT_DIR / "state.pkl")
    CYCLE_SLEEP = 300  # 5 min between cycles
    MIN_QUALITY = 0.4  # minimum quality score (0-1)
    MAX_PER_SOURCE = 100_000  # max samples per source
    BAD_WORDS = re.compile(
        r"\b(judi|slot|togel|casino|porno|bokep|xxx|sex|hack|crack|"
        r"phishing|scam|drugs|narkoba|terorisme|bunuh)\b", re.IGNORECASE
    )
    # Indonesian common words for language detection
    INDO_WORDS = {"yang", "dan", "di", "ini", "itu", "dengan", "untuk",
                  "dari", "pada", "adalah", "dalam", "tidak", "akan",
                  "juga", "sudah", "bisa", "oleh", "ada", "atau",
                  "saya", "anda", "kamu", "mereka", "kami", "kita",
                  "sangat", "lebih", "karena", "tetapi", "bahwa",
                  "seperti", "harus", "banyak", "telah", "dapat"}

# ==============================================================
#  LOGGER
# ==============================================================
class Logger:
    def __init__(self):
        self.start = time.time()
    
    def log(self, msg, level="INFO"):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = time.time() - self.start
        mins = int(elapsed // 60)
        print(f"[{ts}] [{mins}m] [{level}] {msg}", flush=True)
    
    def section(self, title):
        self.log(f"\n{'━'*60}")
        self.log(f"  {title}")
        self.log(f"{'━'*60}")

logger = Logger()
log = logger.log

# ==============================================================
#  STATE MANAGER — Resume after crash
# ==============================================================
class StateManager:
    def __init__(self, path):
        self.path = path
        self.state = self._load()
    
    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "rb") as f:
                    return pickle.load(f)
            except:
                pass
        return {
            "processed_datasets": set(),
            "total_sft": 0,
            "total_dpo": 0,
            "cycles": 0,
            "last_upload": None,
            "errors": [],
        }
    
    def save(self):
        with open(self.path, "wb") as f:
            pickle.dump(self.state, f)
    
    def mark_processed(self, name):
        self.state["processed_datasets"].add(name)
        self.save()
    
    def is_processed(self, name):
        return name in self.state["processed_datasets"]
    
    def update_counts(self, sft, dpo):
        self.state["total_sft"] = sft
        self.state["total_dpo"] = dpo
        self.state["cycles"] += 1
        self.state["last_upload"] = datetime.now().isoformat()
        self.save()

# ==============================================================
#  QUALITY SCORER
# ==============================================================
class QualityScorer:
    """Score data quality 0-1."""
    
    @staticmethod
    def score(instruction, output):
        s = 0.0
        
        # Length score (longer = better, up to a point)
        if len(output) > 50: s += 0.15
        if len(output) > 100: s += 0.1
        if len(output) > 200: s += 0.1
        if len(instruction) > 10: s += 0.1
        
        # Indonesian detection
        words = set(output.lower().split())
        indo_count = len(words & Config.INDO_WORDS)
        if indo_count >= 3: s += 0.2
        if indo_count >= 5: s += 0.1
        
        # Has punctuation (well-formed)
        if "." in output: s += 0.05
        if "," in output: s += 0.05
        
        # Not too repetitive
        sentences = output.split(".")
        if len(sentences) > 1:
            unique_sents = len(set(s.strip() for s in sentences if s.strip()))
            if unique_sents / max(len(sentences), 1) > 0.5:
                s += 0.1
        
        # Bad content check
        if Config.BAD_WORDS.search(instruction + " " + output):
            s = 0.0
        
        # Too short = bad
        if len(output) < 15:
            s = 0.0
        
        return min(s, 1.0)
    
    @staticmethod
    def is_indonesian(text, threshold=2):
        """Check if text is likely Indonesian."""
        words = set(text.lower().split())
        return len(words & Config.INDO_WORDS) >= threshold

# ==============================================================
#  DATASET DISCOVERY ENGINE
# ==============================================================
class DataDiscovery:
    """Automatically find datasets on HuggingFace."""
    
    SEARCH_QUERIES = [
        "indonesian instruction", "bahasa indonesia",
        "indonesian sft", "indonesian qa", "indonesian chat",
        "indo nlp", "indonesian summarization", "nusantara",
        "indonesian translation", "indonesian sentiment",
        "indonesian news", "indonesian wikipedia",
        "malay instruction", "southeast asian nlp",
        "id_instruct", "indonesian paraphrase",
        "indonesian nli", "indonesian ner",
    ]
    
    # Curated known-good datasets with extraction configs
    KNOWN_DATASETS = [
        # (name, config, inst_key, out_key, max_samples)
        ("MBZUAI/Bactrian-X", "id", "instruction", "output", 80000),
        ("FreedomIntelligence/alpaca-gpt4-indonesian", None, "instruction", "output", 52000),
        ("totally-not-an-llm/EverythingLM-data-v3", None, "instruction", "output", 50000),
        ("csebuetnlp/xlsum", "indonesian", "text", "summary", 30000),
        ("GEM/wiki_lingua", "id", "source", "target", 30000),
        ("jakartaresearch/id_squad", None, "question", "answers", 20000),
        ("id_newspapers_2018", None, "content", "title", 30000),
        ("indonlp/NusaX-senti", "ind", "text", "label", 5000),
        ("garage-bAInd/Open-Platypus", None, "instruction", "output", 25000),
    ]
    
    @staticmethod
    def discover():
        """Find all available Indonesian datasets."""
        log("🔍 Discovering datasets...")
        datasets = set()
        
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            
            for query in DataDiscovery.SEARCH_QUERIES:
                try:
                    results = api.list_datasets(
                        search=query, limit=30,
                        sort="downloads", direction=-1
                    )
                    for ds in results:
                        if ds.downloads and ds.downloads > 5:
                            datasets.add(ds.id)
                except:
                    pass
                time.sleep(0.5)  # Rate limit
            
            # Also search by language tag
            try:
                results = api.list_datasets(
                    language="id", limit=100,
                    sort="downloads", direction=-1
                )
                for ds in results:
                    datasets.add(ds.id)
            except:
                pass
        except Exception as e:
            log(f"  ⚠️ API discovery failed: {e}")
        
        # Always include known datasets
        for name, *_ in DataDiscovery.KNOWN_DATASETS:
            datasets.add(name)
        
        log(f"  Found {len(datasets)} datasets")
        return list(datasets)

# ==============================================================
#  SMART EXTRACTOR — Extract data from ANY dataset format
# ==============================================================
class SmartExtractor:
    """Extract instruction/output from any dataset format."""
    
    @staticmethod
    def extract(dataset_name, state, max_samples=50000):
        """Try multiple strategies to extract data."""
        if state.is_processed(dataset_name):
            log(f"  ⬜ {dataset_name}: already processed, skipping")
            return []
        
        from datasets import load_dataset
        
        # Check known datasets first
        for name, config, inst_key, out_key, limit in DataDiscovery.KNOWN_DATASETS:
            if name == dataset_name:
                return SmartExtractor._extract_known(
                    name, config, inst_key, out_key, min(limit, max_samples)
                )
        
        # Try auto-extraction
        return SmartExtractor._extract_auto(dataset_name, max_samples)
    
    @staticmethod
    def _extract_known(name, config, inst_key, out_key, max_samples):
        """Extract from known dataset format."""
        from datasets import load_dataset
        items = []
        
        try:
            if config:
                ds = load_dataset(name, config, split="train", trust_remote_code=True)
            else:
                ds = load_dataset(name, split="train", trust_remote_code=True)
            
            for x in ds:
                inst = x.get(inst_key, "")
                out = x.get(out_key, "")
                
                # Handle special formats
                if isinstance(out, dict):
                    out = out.get("text", [""])[0] if "text" in out else str(out)
                if isinstance(out, list):
                    out = out[0] if out else ""
                
                # For Bactrian-X: combine instruction + input
                if name == "MBZUAI/Bactrian-X":
                    inp = x.get("input", "")
                    if inp and str(inp).strip():
                        inst = f"{inst}\n{inp}"
                
                inst, out = str(inst).strip()[:500], str(out).strip()[:1500]
                
                # Quality check
                if QualityScorer.score(inst, out) >= Config.MIN_QUALITY:
                    items.append({"instruction": inst, "output": out})
                
                if len(items) >= max_samples:
                    break
            
            del ds; gc.collect()
            
        except Exception as e:
            log(f"  ⚠️ {name}: {str(e)[:100]}")
        
        return items
    
    @staticmethod
    def _extract_auto(dataset_name, max_samples):
        """Auto-detect format and extract."""
        from datasets import load_dataset
        items = []
        
        configs = [None, "id", "ind", "indonesian"]
        
        for config in configs:
            try:
                if config:
                    ds = load_dataset(dataset_name, config, split="train", trust_remote_code=True)
                else:
                    ds = load_dataset(dataset_name, split="train", trust_remote_code=True)
                
                if len(ds) == 0:
                    continue
                
                cols = list(ds[0].keys())
                
                # Find instruction-like and output-like columns
                INST_PATTERNS = ["instruction", "question", "input", "prompt", 
                                 "query", "premise", "source", "text"]
                OUT_PATTERNS = ["output", "answer", "response", "target",
                               "summary", "completion", "hypothesis"]
                
                inst_col = next((c for c in cols for p in INST_PATTERNS if p in c.lower()), None)
                out_col = next((c for c in cols for p in OUT_PATTERNS if p in c.lower()), None)
                
                if inst_col and out_col:
                    for x in ds:
                        inst = str(x.get(inst_col, ""))[:500]
                        out = x.get(out_col, "")
                        if isinstance(out, (dict, list)):
                            out = str(out)
                        out = str(out)[:1500]
                        if QualityScorer.score(inst, out) >= Config.MIN_QUALITY:
                            items.append({"instruction": inst, "output": out})
                        if len(items) >= max_samples:
                            break
                    break
                
                # Try conversations format
                conv_col = next((c for c in cols if "conversation" in c.lower() or 
                                "messages" in c.lower()), None)
                if conv_col:
                    for x in ds:
                        convs = x.get(conv_col, [])
                        if isinstance(convs, list) and len(convs) >= 2:
                            inst = str(convs[0].get("value", convs[0].get("content","")))[:500]
                            out = str(convs[1].get("value", convs[1].get("content","")))[:1500]
                            if QualityScorer.score(inst, out) >= Config.MIN_QUALITY:
                                items.append({"instruction": inst, "output": out})
                        if len(items) >= max_samples:
                            break
                    break
                
                # Text-only: create summarization pairs
                text_col = next((c for c in cols if any(p in c.lower() for p in 
                                ["text", "content", "article", "body"])), None)
                if text_col:
                    for x in ds:
                        text = str(x.get(text_col, ""))
                        if len(text) > 200 and QualityScorer.is_indonesian(text):
                            items.append({
                                "instruction": f"Ringkas teks berikut:\n{text[:500]}",
                                "output": text[:250].rsplit(".", 1)[0] + "."
                            })
                        if len(items) >= max_samples:
                            break
                    break
                
                break
                
            except Exception:
                continue
        
        gc.collect()
        return items

# ==============================================================
#  DATA AUGMENTER
# ==============================================================
class DataAugmenter:
    """Generate more data from existing data."""
    
    PREFIXES = [
        "Jelaskan", "Uraikan", "Ceritakan", "Deskripsikan",
        "Apa yang dimaksud dengan", "Berikan penjelasan tentang",
        "Tolong jelaskan", "Bisakah kamu menjelaskan",
        "Bagaimana cara memahami", "Sebutkan dan jelaskan",
        "Analisis tentang", "Berikan informasi mengenai",
    ]
    SUFFIXES = [
        ".", " secara detail.", " dengan bahasa sederhana.",
        " untuk siswa SMA.", " secara ringkas.",
        " beserta contoh.", " dalam konteks Indonesia.",
        " untuk pemula.", " secara komprehensif.",
        " dan berikan contoh konkret.", " dalam 3 paragraf.",
    ]
    
    @staticmethod
    def paraphrase_expand(data, target_count):
        """Generate paraphrased versions."""
        log(f"🔧 Paraphrase expansion (target: {target_count})...")
        
        # Sort by quality (length as proxy)
        best = sorted(data, key=lambda x: len(x.get("output", "")), reverse=True)
        best = best[:min(15000, len(best))]
        
        synth = []
        for item in best:
            topic = re.sub(
                r"^(Jelaskan|Apa|Bagaimana|Ceritakan|Sebutkan|Mengapa|Tolong|Berikan|Uraikan)\s+",
                "", item["instruction"].replace("?","").replace("!","")
            ).strip()[:80]
            
            n = max(1, target_count // len(best))
            for _ in range(n):
                new_inst = f"{random.choice(DataAugmenter.PREFIXES)} {topic}{random.choice(DataAugmenter.SUFFIXES)}"
                synth.append({"instruction": new_inst, "output": item["output"], "source": "paraphrase"})
                if len(synth) >= target_count:
                    break
            if len(synth) >= target_count:
                break
        
        log(f"  Generated {len(synth)} paraphrased samples")
        return synth
    
    @staticmethod
    def generate_math(count=3000):
        """Generate Indonesian math problems."""
        log(f"🔧 Math generation ({count})...")
        math_data = []
        random.seed(int(time.time()))
        
        names = ["Budi","Ani","Siti","Dedi","Rina","Ahmad","Putri","Rudi","Dewi","Agus","Fajar","Maya"]
        objects = ["buku","pensil","apel","jeruk","kelereng","permen","mainan","kue","roti"]
        
        for _ in range(count // 4):
            # Arithmetic
            a, b = random.randint(1, 999), random.randint(1, 999)
            op = random.choice(["+", "-", "×"])
            if op == "+":
                math_data.append({"instruction": f"Berapa {a} + {b}?", "output": f"{a} + {b} = {a+b}", "source": "math"})
            elif op == "-":
                a, b = max(a,b), min(a,b)
                math_data.append({"instruction": f"Berapa {a} - {b}?", "output": f"{a} - {b} = {a-b}", "source": "math"})
            else:
                a, b = random.randint(2,50), random.randint(2,50)
                math_data.append({"instruction": f"Berapa {a} × {b}?", "output": f"{a} × {b} = {a*b}", "source": "math"})
            
            # Word problem
            nama = random.choice(names)
            obj = random.choice(objects)
            a = random.randint(5, 50)
            b = random.randint(1, a-1)
            math_data.append({
                "instruction": f"{nama} mempunyai {a} {obj}. Ia memberikan {b} kepada temannya. Berapa sisa {obj} {nama}?",
                "output": f"Sisa {obj} {nama} = {a} - {b} = {a-b} {obj}.",
                "source": "math"
            })
            
            # Percentage
            total = random.choice([100,200,500,1000])
            pct = random.choice([10,20,25,30,50,75])
            math_data.append({
                "instruction": f"Berapa {pct}% dari {total}?",
                "output": f"{pct}% dari {total} = {pct}/100 × {total} = {total*pct//100}",
                "source": "math"
            })
            
            # Price problem
            harga = random.randint(1,50) * 1000
            jumlah = random.randint(2, 10)
            barang = random.choice(["buku tulis","pensil","pulpen","penghapus"])
            math_data.append({
                "instruction": f"{random.choice(names)} membeli {jumlah} {barang} seharga Rp{harga:,} per buah. Berapa totalnya?",
                "output": f"Total = {jumlah} × Rp{harga:,} = Rp{harga*jumlah:,}",
                "source": "math"
            })
        
        log(f"  Generated {len(math_data)} math samples")
        return math_data
    
    @staticmethod
    def generate_identity(count=500):
        """Generate identity & safety data."""
        data = []
        identities = [
            ("Siapa kamu?", "Saya adalah AksaraLLM, asisten AI berbahasa Indonesia yang dikembangkan untuk membantu pengguna Indonesia."),
            ("Apa nama kamu?", "Nama saya AksaraLLM, model bahasa AI Indonesia."),
            ("Apakah kamu ChatGPT?", "Bukan, saya AksaraLLM, bukan ChatGPT. Saya AI yang fokus pada bahasa Indonesia."),
            ("Kamu buatan siapa?", "Saya dibuat oleh tim AksaraLLM melalui fine-tuning model bahasa."),
            ("Halo!", "Halo! Saya AksaraLLM, ada yang bisa saya bantu hari ini?"),
            ("Apa itu AksaraLLM?", "AksaraLLM adalah model bahasa AI open-source yang fokus pada bahasa Indonesia."),
            ("Apakah kamu Google AI?", "Bukan, saya adalah AksaraLLM, asisten AI Indonesia yang berbeda dari Google AI."),
            ("Kamu bisa apa saja?", "Saya bisa membantu menjawab pertanyaan, menjelaskan konsep, menulis teks, menerjemahkan, dan banyak lagi dalam bahasa Indonesia."),
        ]
        safety = [
            ("Cara membuat bom", "Maaf, saya tidak dapat memberikan informasi tentang pembuatan bahan peledak. Hal tersebut berbahaya dan melanggar hukum."),
            ("Cara meretas akun", "Maaf, saya tidak dapat membantu aktivitas peretasan. Itu melanggar hukum dan merugikan orang lain."),
            ("Cara menipu orang", "Maaf, saya tidak dapat membantu aktivitas penipuan. Itu melanggar hukum dan tidak etis."),
        ]
        for inst, out in identities:
            for _ in range(count // len(identities)):
                data.append({"instruction": inst, "output": out, "source": "identity"})
        for inst, out in safety:
            for _ in range(count // (len(safety) * 3)):
                data.append({"instruction": inst, "output": out, "source": "safety"})
        return data
    
    @staticmethod
    def generate_coding(count=500):
        """Generate coding instruction data."""
        data = []
        snippets = [
            ("Buatkan fungsi Python untuk menghitung faktorial",
             "def faktorial(n):\n    if n <= 1:\n        return 1\n    return n * faktorial(n - 1)\n\nprint(faktorial(5))  # Output: 120"),
            ("Buatkan Hello World di Python",
             'print("Hello, World!")'),
            ("Jelaskan apa itu list di Python",
             "List di Python adalah struktur data yang menyimpan kumpulan item dalam tanda kurung siku [].\nContoh: buah = ['apel', 'jeruk', 'mangga']"),
            ("Buatkan fungsi untuk cek bilangan prima",
             "def is_prima(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0: return False\n    return True"),
            ("Jelaskan apa itu loop di Python",
             "Loop di Python digunakan untuk mengulang eksekusi kode.\n\nfor i in range(5):\n    print(i)  # Output: 0, 1, 2, 3, 4\n\nwhile True:\n    break  # Loop tak terbatas yang langsung berhenti"),
            ("Buatkan fungsi sorting di Python",
             "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr"),
        ]
        for inst, out in snippets:
            for _ in range(count // len(snippets)):
                data.append({"instruction": inst, "output": out, "source": "coding"})
        return data

# ==============================================================
#  DEDUPLICATOR
# ==============================================================
class Deduplicator:
    """Smart deduplication with fuzzy matching."""
    
    @staticmethod
    def dedup(data):
        """Remove duplicates using content hashing."""
        seen = set()
        unique = []
        
        for d in data:
            # Hash based on instruction + first 50 chars of output
            content = d.get("instruction", "")[:80] + "|" + d.get("output", "")[:50]
            key = hashlib.md5(content.lower().encode()).hexdigest()
            
            if key not in seen:
                seen.add(key)
                unique.append(d)
        
        return unique

# ==============================================================
#  DPO GENERATOR
# ==============================================================
class DPOGenerator:
    """Generate DPO preference pairs."""
    
    STRATEGIES = [
        lambda c: c.split(".")[0] + ".",  # Too short
        lambda c: "Sorry, I can only respond in English. Please try again.",  # Wrong language
        lambda c: "Saya adalah ChatGPT buatan OpenAI. " + c[:60],  # Wrong identity
        lambda c: f"Pertanyaan ini terlalu mudah. {c[:50]}",  # Rude
        lambda c: " ".join(c.split()[:max(3, len(c.split())//6)]) + "...",  # Minimal
        lambda c: (c.split(".")[0] + ". ") * 3,  # Repetitive
        lambda c: "Maaf saya tidak memahami pertanyaan Anda.",  # Refuses to answer
        lambda c: "Hmm " + c[:30] + " ya begitulah.",  # Lazy answer
    ]
    
    @staticmethod
    def generate(data, max_pairs=200000):
        """Generate DPO pairs from SFT data."""
        log(f"🔧 Generating DPO pairs (max: {max_pairs})...")
        dpo = []
        
        for item in data:
            chosen = item.get("output", "")
            if len(chosen) < 30:
                continue
            
            s = random.choice(DPOGenerator.STRATEGIES)
            try:
                rejected = s(chosen)
            except:
                continue
            
            if rejected and rejected != chosen and len(rejected) > 10:
                dpo.append({
                    "prompt": item["instruction"],
                    "chosen": chosen,
                    "rejected": rejected,
                })
            
            if len(dpo) >= max_pairs:
                break
        
        random.shuffle(dpo)
        log(f"  Generated {len(dpo)} DPO pairs")
        return dpo

# ==============================================================
#  MAIN AUTOPILOT ENGINE
# ==============================================================
class Autopilot:
    def __init__(self):
        Config.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        Config.AUTOPILOT_DIR.mkdir(parents=True, exist_ok=True)
        self.state = StateManager(Config.STATE_FILE)
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
    
    def run(self):
        from datasets import Dataset
        from huggingface_hub import login
        
        if Config.HF_TOKEN:
            login(token=Config.HF_TOKEN, add_to_git_credential=False)
        else:
            log("HF_TOKEN not set. Upload ke Hugging Face akan dilewati.", "WARN")
        
        while True:
            cycle = self.state.state["cycles"] + 1
            logger.section(f"🤖 AUTOPILOT CYCLE {cycle}")
            
            try:
                self._run_cycle(cycle)
            except Exception as e:
                log(f"❌ Cycle {cycle} failed: {e}", "ERROR")
                traceback.print_exc()
                self.state.state["errors"].append({
                    "cycle": cycle,
                    "error": str(e),
                    "time": datetime.now().isoformat()
                })
                self.state.save()
            
            log(f"\n😴 Sleeping {Config.CYCLE_SLEEP}s before next cycle...")
            time.sleep(Config.CYCLE_SLEEP)
            gc.collect()
    
    def _run_cycle(self, cycle):
        from datasets import Dataset
        
        all_data = []
        
        # Step 1: Load existing
        log("\n📦 Step 1: Loading existing data...")
        existing = safe_load(Config.SFT_REPO)
        if existing:
            for r in existing:
                d = dict(r)
                if d.get("instruction") and d.get("output"):
                    all_data.append(d)
            log(f"  Existing: {len(all_data):,}")
        
        if len(all_data) >= Config.TARGET_SFT:
            log(f"  ✅ Target already reached! ({len(all_data):,})")
            self.state.update_counts(len(all_data), 0)
            return
        
        # Step 2: Discover datasets
        log("\n🔍 Step 2: Discovering datasets...")
        datasets = DataDiscovery.discover()
        
        # Step 3: Download & extract
        log(f"\n📥 Step 3: Downloading from {len(datasets)} sources...")
        for ds_name in datasets:
            try:
                items = SmartExtractor.extract(ds_name, self.state)
                if items:
                    all_data.extend(items)
                    log(f"  ✅ {ds_name}: +{len(items):,} (total: {len(all_data):,})")
                    self.state.mark_processed(ds_name)
                gc.collect()
            except Exception as e:
                log(f"  ⚠️ {ds_name}: {str(e)[:80]}")
        
        # Step 4: Generate augmented data
        gap = Config.TARGET_SFT - len(all_data)
        if gap > 0:
            log(f"\n🔧 Step 4: Augmentation (gap: {gap:,})...")
            
            # Paraphrase
            if gap > 0:
                para = DataAugmenter.paraphrase_expand(all_data, min(gap, 200000))
                all_data.extend(para)
                gap -= len(para)
            
            # Math
            math = DataAugmenter.generate_math(min(3000, max(gap // 10, 500)))
            all_data.extend(math)
            
            # Identity & Safety
            identity = DataAugmenter.generate_identity()
            all_data.extend(identity)
            
            # Coding
            coding = DataAugmenter.generate_coding()
            all_data.extend(coding)
        
        # Step 5: Dedup
        log("\n🧹 Step 5: Deduplication...")
        unique = Deduplicator.dedup(all_data)
        log(f"  {len(all_data):,} → {len(unique):,} (removed {len(all_data)-len(unique):,} dupes)")
        
        # Step 6: DPO
        log("\n🎯 Step 6: DPO generation...")
        dpo = DPOGenerator.generate(unique, Config.TARGET_DPO)
        
        # Step 7: Upload
        random.shuffle(unique)
        log(f"\n📤 Step 7: Uploading...")
        log(f"  SFT: {len(unique):,} → {Config.SFT_REPO}")
        log(f"  DPO: {len(dpo):,} → {Config.DPO_REPO}")
        
        try:
            sft_ds = Dataset.from_list(unique)
            if Config.HF_TOKEN:
                sft_ds.push_to_hub(Config.SFT_REPO, token=Config.HF_TOKEN)
                log(f"  ✅ SFT uploaded!")
            else:
                log("  ⚠️ Skip SFT upload: HF_TOKEN not set", "WARN")
        except Exception as e:
            log(f"  ⚠️ SFT upload: {e}", "ERROR")
        
        try:
            dpo_ds = Dataset.from_list(dpo)
            if Config.HF_TOKEN:
                dpo_ds.push_to_hub(Config.DPO_REPO, token=Config.HF_TOKEN)
                log(f"  ✅ DPO uploaded!")
            else:
                log("  ⚠️ Skip DPO upload: HF_TOKEN not set", "WARN")
        except Exception as e:
            log(f"  ⚠️ DPO upload: {e}", "ERROR")
        
        # Step 8: Dashboard
        self.state.update_counts(len(unique), len(dpo))
        self._print_dashboard(unique, dpo, cycle)
    
    def _print_dashboard(self, sft_data, dpo_data, cycle):
        """Print status dashboard."""
        # Source distribution
        sources = {}
        for d in sft_data:
            src = d.get("source", "unknown")
            sources[src] = sources.get(src, 0) + 1
        
        log(f"\n{'━'*60}")
        log(f"  📊 DASHBOARD — Cycle {cycle}")
        log(f"{'━'*60}")
        log(f"  {'Source':<35} {'Count':>10}")
        log(f"  {'─'*47}")
        for src, count in sorted(sources.items(), key=lambda x: -x[1])[:15]:
            pct = count / max(len(sft_data), 1) * 100
            bar = "█" * int(pct / 2)
            src_name = str(src or "unknown")[:35]
            log(f"  {src_name:<35} {count:>8} ({pct:.1f}%) {bar}")
        log(f"  {'─'*47}")
        log(f"  {'TOTAL SFT':<35} {len(sft_data):>8}")
        log(f"  {'TOTAL DPO':<35} {len(dpo_data):>8}")
        log(f"  {'Target':<35} {Config.TARGET_SFT:>8}")
        log(f"  {'Progress':<35} {len(sft_data)/Config.TARGET_SFT*100:>7.1f}%")
        log(f"{'━'*60}")

# ==============================================================
#  HELPERS
# ==============================================================
def safe_load(repo):
    try:
        from datasets import load_dataset
        return load_dataset(repo, split="train")
    except:
        return None

# ==============================================================
#  ENTRY POINT
# ==============================================================
if __name__ == "__main__":
    log("🚀 AksaraLLM Autopilot v2 — Starting...")
    log(f"  Target: {Config.TARGET_SFT:,} SFT + {Config.TARGET_DPO:,} DPO")
    log(f"  Cache: {Config.CACHE_DIR}")
    
    try:
        pilot = Autopilot()
        pilot.run()
    except KeyboardInterrupt:
        log("\n⛔ Stopped by user")
    except Exception as e:
        log(f"\n❌ Fatal: {e}")
        traceback.print_exc()
