# Quick Start

## 1. Pakai model langsung via Transformers

Kalau tujuan Anda hanya mencoba model, ini jalur paling pendek:

```bash
pip install -U transformers accelerate torch
```

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "AksaraLLM/aksarallm-1.5b-v2"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
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

## 2. Pakai tool repo ini

Kalau Anda ingin CLI dan Web UI yang sudah disiapkan repo:

```bash
pip install -e .
```

Base install di atas sengaja ringan untuk docs, test, dan review repo.

Untuk CLI inference:

```bash
pip install -e ".[runtime]"
```

Untuk Web UI Gradio:

```bash
pip install -e ".[demo]"
```

Untuk smoke test publik:

```bash
python scripts/release_check.py
```

## 3. Chat via terminal

```bash
aksarallm-chat --model AksaraLLM/aksarallm-1.5b-v2
```

Fitur yang tersedia:

- `/reset` untuk hapus riwayat percakapan
- `/history` untuk lihat ringkasan turn terakhir
- `/exit` untuk keluar

Mode sekali jalan:

```bash
aksarallm-chat --message "Apa itu Pancasila?"
```

## 4. Chat via Web UI

```bash
aksarallm-webui --model AksaraLLM/aksarallm-1.5b-v2
```

Atau dari checkout repo:

```bash
python app.py
```

Lalu buka `http://localhost:7860`.

## 5. Ekspektasi yang realistis

- CPU tetap bisa dipakai, tetapi respons model 1.5B akan lebih lambat.
- Untuk pengalaman yang lebih enak, gunakan GPU atau Apple Silicon yang memadai.
- Sebelum dipakai publik, cek dulu [release standard](./release_standard.md) supaya tidak berhenti di model card dan demo saja.
- Jika Anda ingin upload artifact atau hasil evaluasi ke Hugging Face, set `HF_TOKEN` lewat environment variable, bukan di source code.
