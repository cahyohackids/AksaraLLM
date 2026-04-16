---
language:
- id
- en
license: apache-2.0
library_name: transformers
pipeline_tag: text-generation
tags:
- indonesian
- aksarallm
- bahasa-indonesia
- qwen2
- sft
- dpo
- chat
base_model: Qwen/Qwen2.5-1.5B-Instruct
datasets:
- AksaraLLM/aksara-mega-sft-v5
- AksaraLLM/aksara-dpo-id-v4
model-index:
- name: aksarallm-1.5b-v2
  results: []
---

# AksaraLLM 1.5B v2

Model chat bahasa Indonesia berbasis `Qwen2.5-1.5B-Instruct`, ditujukan untuk eksperimen, demo, dan iterasi produk yang fokus pada Bahasa Indonesia.

## Yang Sudah Siap Dipakai

- Inference langsung via `transformers`
- CLI interaktif untuk chat lokal
- Web UI Gradio dari repo ini
- Konfigurasi model yang lebih konsisten antara docs, CLI, dan demo
- Release smoke check untuk jalur publik dasar

## Quick Start

### Pakai model langsung via Transformers

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "AksaraLLM/aksarallm-1.5b-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True,
)

messages = [
    {"role": "system", "content": "Kamu adalah AksaraLLM, asisten AI berbahasa Indonesia yang membantu."},
    {"role": "user", "content": "Jelaskan apa itu Pancasila."},
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.7, top_p=0.9)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

### Pakai CLI dan Web UI repo

```bash
pip install -e .
```

Kalau Anda ingin menjalankan CLI inference lokal:

```bash
pip install -e ".[runtime]"
```

Kalau Anda ingin Web UI Gradio:

```bash
pip install -e ".[demo]"
```

CLI:

```bash
aksarallm-chat --model AksaraLLM/aksarallm-1.5b-v2
```

Web UI:

```bash
aksarallm-webui --model AksaraLLM/aksarallm-1.5b-v2
```

Atau jalankan langsung dari checkout repo:

```bash
python app.py
```

## Batas Praktis

- CPU tetap bisa dipakai, tetapi akan terasa lambat untuk penggunaan harian.
- Base install repo sengaja dibuat ringan; dependency berat dipindah ke extra runtime/demo.
- Untuk demo publik, siapkan evaluasi dan guardrail terpisah; repo ini baru menyiapkan jalur pemakaian dasar.
- Artifact GGUF dan rilis Ollama perlu dianggap sebagai jalur distribusi terpisah yang harus diverifikasi per release.

## Model Details

| Attribute | Value |
|---|---|
| Base Model | Qwen2.5-1.5B-Instruct |
| Parameters | 1.78B |
| Context Window | 32K tokens |
| Training Data | 500K SFT + 200K DPO |
| Training Hardware | Google Cloud TPU v6e-4 |
| Training Method | Full fine-tuning (SFT → DPO) |
| Precision | BFloat16 |
| License | Apache 2.0 |

## Training Data

### SFT Data

| Sumber | Jumlah | Kategori |
|---|---|---|
| Bactrian-X Indonesian | ~80K | Instruksi umum |
| Alpaca GPT-4 Indonesian | ~52K | Instruksi umum |
| XLSum Indonesian | ~30K | Summarization |
| WikiLingua ID | ~30K | Cross-lingual |
| Paraphrase Augmentation | ~200K | Augmented |
| Math & Coding | ~3.5K | STEM |
| Identity & Safety | ~500 | Alignment |
| Other Sources | ~100K | Mixed |

### DPO Data

Rejected response dibangun dari pola yang sengaja tidak diinginkan, misalnya:

- terlalu pendek
- salah bahasa
- salah identitas
- kasar
- malas menjawab
- repetitif
- menolak tanpa alasan
- asal-asalan

## Evaluation Status

Skor benchmark publik yang stabil belum dicantumkan di README ini. Untuk standar rilis yang lebih aman, gunakan checklist di [docs/release_standard.md](./docs/release_standard.md) agar artifact training, evaluasi, dan jalur pakai tetap sinkron.

## Limitations

- Bukan pengganti profesional untuk domain medis, hukum, atau keuangan
- Masih bisa berhalusinasi
- Performa multi-turn panjang tetap perlu diuji per use case
- Belum ada klaim dukungan bahasa daerah yang kuat

## Repo Guide

- Panduan cepat: [docs/quickstart.md](./docs/quickstart.md)
- Standar rilis: [docs/release_standard.md](./docs/release_standard.md)
- Kebijakan keamanan: [SECURITY.md](./SECURITY.md)
- Release gate lokal: `python scripts/release_check.py`

## License

Apache License 2.0
