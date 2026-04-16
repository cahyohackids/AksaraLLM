# Release Standard: "Udah Enak Dipakai Orang"

Dokumen ini adalah bar minimal supaya hasil training tidak berhenti di status "model jadi", tetapi naik ke status "orang baru bisa install, mencoba, memahami batasannya, dan memberi feedback tanpa bingung".

## Definisi Sederhana

Sebuah release dianggap "udah enak dipakai orang" kalau:

- orang baru bisa menjalankan model dalam kurang dari 30 menit
- jalur CLI dan/atau Web UI jelas dan tidak patah di langkah pertama
- nama artifact, docs, dan model id konsisten
- keterbatasan model dijelaskan dengan jujur
- ada bukti evaluasi minimum untuk use case yang dipromosikan

## Checklist Wajib

### 1. Jalur pakai harus nyata

- ada satu quickstart untuk `transformers`
- ada satu jalur CLI yang jelas
- ada satu jalur Web UI yang jelas
- semua command di docs cocok dengan file dan entrypoint yang benar

### 2. Artifact harus konsisten

- nama model final, checkpoint, tokenizer, dan dataset sinkron
- README, script upload, dan config menunjuk ke repo/model id yang sama
- versi release punya changelog singkat: apa yang berubah, apa yang belum

### 3. Ekspektasi hardware harus jujur

- tulis apakah CPU hanya untuk coba-coba atau memang nyaman dipakai
- jelaskan perangkat yang direkomendasikan
- sebutkan limit konteks dan latency praktis bila sudah diketahui

### 4. Evaluasi minimum sebelum publik

- identity check
- refusal/safety check dasar
- Indonesian general QA check
- formatting/obedience check
- satu benchmark atau suite internal yang relevan dengan use case utama

Kalau skor formal belum siap, minimal ada hasil uji manual yang terdokumentasi dan tanggal verifikasinya jelas.

### 5. UX dan failure mode

- error karena dependency kurang harus memberi pesan yang jelas
- reset percakapan harus mudah
- contoh prompt harus disediakan
- user tidak perlu membaca source code untuk mulai mencoba

### 6. Feedback loop

- sediakan tempat melaporkan issue atau hasil uji
- bedakan antara bug inference, masalah kualitas jawaban, dan masalah packaging

## Gate Sebelum Menyebut "Public/Beta"

Sebelum sebuah model disebut `public`, `beta`, atau `siap dipakai`, pastikan:

- quickstart sudah dicoba dari environment bersih
- CLI dan Web UI sudah benar-benar jalan
- README tidak menjanjikan artifact yang belum ada
- link model, dataset, dan evaluator tidak saling silang
- satu orang lain selain pembuat utama bisa mengikuti docs tanpa bantuan langsung

## Hubungan Dengan Training

Untuk model training `1.5B`, standar ini berarti:

- output training harus punya nama akhir yang tegas
- jalur dari checkpoint ke artifact user-facing harus terdokumentasi
- evaluasi harus dipasang sebagai gate rilis, bukan pekerjaan setelah publish
- demo harus memakai artifact yang sama dengan yang dipromosikan

## Exit Criteria

Kalau masih ada salah satu kondisi di bawah ini, release belum layak disebut "udah enak dipakai orang":

- demo patah saat start
- docs menunjuk command yang tidak ada
- model id di README beda dengan model id yang benar-benar dipakai
- hasil evaluasi belum jelas tapi marketing copy terlalu tinggi
- orang baru masih harus tanya pembuatnya untuk langkah dasar
