# Security Policy

## Supported Scope

Repo ini menerima laporan untuk:

- secret atau token yang tidak sengaja masuk ke source
- command atau script yang berpotensi mengunggah data tanpa kontrol yang jelas
- jalur install/run yang mengeksekusi resource eksternal tanpa penjelasan
- kelemahan packaging atau release process yang bisa membahayakan user

## Reporting

Jangan buka issue publik untuk secret aktif atau kredensial yang masih berlaku.

Laporkan dengan:

1. deskripsi singkat masalah
2. file/path yang terdampak
3. langkah reproduksi minimum
4. dampak praktis

Jika laporan berisi secret aktif, lakukan rotasi secret tersebut secepat mungkin sebelum atau saat laporan dikirim.

## Baseline Repo Hygiene

Sebelum release publik:

- tidak ada token hardcoded di source
- upload ke Hugging Face harus memakai environment variable
- script user-facing harus tetap memberi help/usage walau dependency runtime belum lengkap
- smoke test dan release check harus lolos
