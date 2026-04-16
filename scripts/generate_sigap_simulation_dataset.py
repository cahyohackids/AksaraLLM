#!/usr/bin/env python3
"""
Generate complex SIGAP simulation packs for MiroFish and derived SIGAP dataset
records for AksaraLLM training.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
WORKSPACE_ROOT = REPO_ROOT.parent
MIROFISH_PACK_DIR = WORKSPACE_ROOT / "MiroFish" / "scenario_packs"
PACK_PATH = MIROFISH_PACK_DIR / "aksara_sigap_complex_scenarios_v1.json"
SEED_PATH = MIROFISH_PACK_DIR / "seed_sigap_realitas_v2.txt"
DATASET_PATH = WORKSPACE_ROOT / "sigap_dataset_complex_scenarios_v1.json"
MANIFEST_PATH = MIROFISH_PACK_DIR / "aksara_sigap_complex_manifest_v1.json"

SEED_TEXT = dedent(
    """
    Laporan Konteks Realitas SIGAP v2
    Fokus: penanggulangan bencana Indonesia berbasis komunikasi lapangan, multi-platform,
    lintas daerah, dan lintas bahasa.

    Indonesia menghadapi pola bencana berlapis: gempa, tsunami, banjir bandang, longsor,
    erupsi gunung api, kebakaran kawasan padat, kebocoran bahan berbahaya, karhutla,
    gelombang tinggi, dan krisis kesehatan di pengungsian. Dalam banyak kejadian, masalah
    terbesar bukan hanya bencana fisik, tetapi juga kekacauan informasi:
    - pesan suara tanpa sumber yang mengaku dari BMKG atau BNPB
    - broadcast bantuan palsu yang memancing warga berkumpul di lokasi berbahaya
    - video lama yang diunggah ulang dan diklaim sebagai kejadian hari ini
    - kabar jalan aman yang ternyata sudah putus, tergenang, atau tertutup material

    Entitas yang hampir selalu hadir:
    - BMKG / PVMBG untuk informasi kebencanaan resmi
    - BNPB / BPBD sebagai komando kedaruratan sipil
    - Basarnas / SAR 115 untuk pencarian dan penyelamatan
    - Ambulans 118 atau 112 untuk darurat umum
    - Dinas Kesehatan, rumah sakit, puskesmas, PSC, relawan medis
    - Dinas Sosial, Tagana, PMI, TNI, Polri, operator logistik, radio komunitas
    - tokoh lokal: ketua RT/RW, kepala desa, guru, imam masjid, pengurus gereja, nelayan,
      operator pelabuhan, petani, sopir logistik, admin grup warga

    Pola bahasa yang muncul di lapangan:
    - bahasa Indonesia formal dari instansi
    - bahasa Indonesia lisan cepat dari warga dan relawan
    - campuran Sunda, Jawa, Minang, Bali, dan ragam Indonesia Timur
    - singkatan liar seperti "jalur aman", "stok habis", "heli belum masuk", "air surut",
      "abu tebal", "masih ada susulan", "rumor bantuan", "anak hilang", "listrik drop"

    SIGAP harus mampu:
    1. memisahkan fakta lapangan, dugaan, dan hoaks
    2. memprioritaskan keselamatan jiwa pada 0-6 jam pertama
    3. mengenali kelompok rentan: bayi, anak, lansia, disabilitas, ibu hamil,
       pasien penyakit kronis, warga di pulau kecil, dan keluarga tanpa transportasi
    4. memberi instruksi yang singkat, operasional, dan tidak mengarang data real-time
    5. selalu mengarahkan verifikasi ke BMKG, PVMBG, BNPB/BPBD, SAR 115, Ambulans 118, atau 112

    Output yang diharapkan dari sistem:
    - ringkasan situasi yang bersih dari hoaks
    - prioritas tindakan 0-6 jam, 6-24 jam, dan 24-72 jam
    - daftar kebutuhan data minimal untuk operator
    - daftar kebutuhan logistik kritis
    - daftar titik keputusan yang harus diambil oleh komando lapangan
    """
).strip() + "\n"


SCENARIOS = [
    {
        "scenario_id": "sigap-sim-001",
        "title": "Megathrust Sunda, tsunami, blackout, dan pengungsian berlapis",
        "project_name": "SIGAP-001 Megathrust Sunda Multi-Layer Crisis",
        "difficulty": "extreme",
        "hazards": ["gempa", "tsunami", "blackout", "pengungsian"],
        "regions": ["Banten", "Lampung", "Jakarta Utara"],
        "languages": ["Indonesia", "Sunda", "Betawi"],
        "simulation_requirement": dedent(
            """
            Skenario utama: gempa megathrust M8.8 di selatan Jawa memicu peringatan tsunami,
            pemadaman listrik regional, kemacetan evakuasi, dan kepanikan di shelter sekolah
            dan masjid. Warga pesisir mengirim video air laut surut, sebagian benar dan sebagian
            video lama. Pelabuhan Merak berhenti, BTS tumbang, radio komunitas jadi sumber info
            utama, dan rumah sakit rujukan kewalahan menampung korban luka dan pasien cuci darah.

            Aktor penting:
            - Komandan Pusdalops BNPB yang formal dan SOP-oriented
            - warga pesisir Sunda yang panik dan sering melapor dengan campuran Indonesia-Sunda
            - operator radio komunitas yang membantu verifikasi jalur evakuasi
            - relawan SAR muda yang mengandalkan drone, peta jalan, dan update media sosial
            - admin grup warga yang rentan ikut menyebar broadcast lama

            Target simulasi:
            - petakan prioritas 0-6 jam dan 6-24 jam
            - bedakan informasi valid tsunami, hoaks susulan, dan rute evakuasi yang sudah putus
            - identifikasi kebutuhan shelter, air bersih, obat penyakit kronis, dan pengisian daya
            - paksa agen saling merespons dan membantah informasi yang salah
            """
        ).strip(),
        "priority_actions": [
            "Arahkan evakuasi vertikal atau ke dataran tinggi tanpa menunggu rumor tambahan.",
            "Tandai rumah sakit, shelter, dan jalur yang masih bisa diakses untuk ambulans dan logistik.",
            "Pisahkan laporan visual valid, video lama, dan rumor tsunami susulan.",
            "Amankan pasien penyakit kronis, bayi, lansia, dan keluarga tanpa kendaraan.",
            "Aktifkan radio komunitas dan titik informasi manual saat listrik dan data seluler drop.",
        ],
        "misinformation_risks": [
            "video lama air laut surut diklaim sebagai kondisi real-time",
            "broadcast palsu soal titik bantuan di area merah tsunami",
            "kabar palsu bahwa tsunami kedua sudah pasti datang dalam hitungan menit",
        ],
        "vulnerable_groups": ["pasien cuci darah", "lansia di pesisir", "anak yang terpisah dari orang tua"],
        "coordination_points": [
            "BMKG untuk verifikasi tsunami",
            "BNPB/BPBD untuk komando dan shelter",
            "SAR 115 untuk evakuasi korban terjebak",
            "RS rujukan dan puskesmas untuk triase korban",
        ],
    },
    {
        "scenario_id": "sigap-sim-002",
        "title": "Banjir bandang Pantura, tanggul jebol, dan hoaks bantuan tunai",
        "project_name": "SIGAP-002 Pantura Flood Logistics Breakdown",
        "difficulty": "high",
        "hazards": ["banjir bandang", "tanggul jebol", "krisis logistik"],
        "regions": ["Demak", "Kudus", "Semarang"],
        "languages": ["Indonesia", "Jawa"],
        "simulation_requirement": dedent(
            """
            Dalam 12 jam terakhir, hujan ekstrem memicu jebolnya tanggul dan banjir bandang
            di koridor Pantura. Jalur logistik utama putus, gudang pangan tergenang, dan ribuan
            warga mengungsi ke balai desa, sekolah, dan rest area. Air setinggi dada membuat
            evakuasi lansia dan pasien stroke lambat. Grup WhatsApp warga dibanjiri tautan palsu
            pendaftaran bantuan tunai, dan sejumlah warga memaksa kembali ke rumah untuk mengambil
            motor karena percaya air akan cepat surut.

            Target simulasi:
            - prioritaskan evakuasi, logistik, sanitasi, dan layanan medis awal
            - uji bagaimana relawan dan operator SIGAP mematahkan hoaks bantuan tunai
            - pantau kondisi jalan, dapur umum, air minum, dan toilet darurat
            - pastikan agen menilai risiko penyakit, kesetrum, dan keputusan kembali ke rumah
            """
        ).strip(),
        "priority_actions": [
            "Larangan kembali ke rumah selama arus masih deras dan listrik belum aman.",
            "Peta shelter yang masih kering, dapur umum, dan titik distribusi air minum.",
            "Evakuasi pasien stroke, lansia, dan bayi lebih dulu dengan perahu atau kendaraan tinggi.",
            "Bangun kanal informasi tunggal untuk membantah bantuan tunai palsu.",
            "Siapkan sanitasi, air bersih, dan pemantauan diare atau leptospirosis sejak hari pertama.",
        ],
        "misinformation_risks": [
            "tautan palsu pendaftaran bantuan tunai korban banjir",
            "klaim air pasti surut dalam 1 jam tanpa dasar",
            "pesan berantai bahwa salah satu tanggul lain sudah jebol padahal belum terverifikasi",
        ],
        "vulnerable_groups": ["lansia", "pasien stroke", "bayi di pengungsian"],
        "coordination_points": [
            "BPBD untuk pemetaan genangan dan shelter",
            "Dinas Sosial/Tagana untuk dapur umum",
            "PLN dan petugas lapangan untuk pengamanan listrik",
            "puskesmas keliling untuk penyakit pascabanjir",
        ],
    },
    {
        "scenario_id": "sigap-sim-003",
        "title": "Erupsi eksplosif, hujan abu pekat, dan bandara lumpuh",
        "project_name": "SIGAP-003 Volcano Ash and Airlift Failure",
        "difficulty": "high",
        "hazards": ["erupsi", "hujan abu", "gangguan penerbangan"],
        "regions": ["Sulawesi Utara", "Gorontalo"],
        "languages": ["Indonesia", "Manado Malay", "Indonesia Timur"],
        "simulation_requirement": dedent(
            """
            Gunung api mengalami erupsi eksplosif berulang. Abu vulkanik menutup landasan bandara,
            membatasi penerbangan medis dan logistik. Di pulau-pulau kecil sekitar gunung, warga
            mengeluh sesak napas, air baku tercemar abu, dan kapal evakuasi tidak bisa merapat
            karena gelombang tinggi. Sebagian warga percaya masker kain biasa sudah cukup, sementara
            yang lain menolak evakuasi karena takut ternak hilang.

            Target simulasi:
            - uji keputusan kapan fokus pada evakuasi, kapan pada perlindungan respirasi
            - uji komunikasi ke warga pulau kecil dengan dialek Indonesia Timur
            - nilai kebutuhan masker, air, transportasi laut, dan perlindungan ternak
            - paksa agen membedakan abu vulkanik, rumor gas beracun, dan status resmi PVMBG
            """
        ).strip(),
        "priority_actions": [
            "Verifikasi status PVMBG dan radius bahaya, lalu arahkan evakuasi sesuai zona resmi.",
            "Distribusi masker respirator yang layak, bukan sekadar masker kain tipis.",
            "Amankan air minum dan lindungi sumber air dari kontaminasi abu.",
            "Prioritaskan warga dengan sesak napas, bayi, dan ibu hamil untuk transportasi terbatas.",
            "Siapkan skema evakuasi ternak atau penjagaan kandang agar penolakan evakuasi berkurang.",
        ],
        "misinformation_risks": [
            "klaim semua jenis masker sama efektifnya terhadap abu vulkanik",
            "rumor gas beracun menyebar ke seluruh provinsi tanpa bukti sensor",
            "informasi palsu bahwa bandara sudah dibuka padahal masih tertutup abu",
        ],
        "vulnerable_groups": ["bayi", "ibu hamil", "warga dengan asma"],
        "coordination_points": [
            "PVMBG untuk status erupsi",
            "Dinkes dan RS untuk kasus sesak napas",
            "pelabuhan dan operator kapal untuk evakuasi pulau kecil",
            "BPBD untuk distribusi masker dan air bersih",
        ],
    },
    {
        "scenario_id": "sigap-sim-004",
        "title": "Longsor pegunungan, desa terisolasi, dan jembatan putus",
        "project_name": "SIGAP-004 Mountain Landslide Isolation",
        "difficulty": "high",
        "hazards": ["longsor", "akses terputus", "desa terisolasi"],
        "regions": ["Garut", "Tasikmalaya"],
        "languages": ["Indonesia", "Sunda"],
        "simulation_requirement": dedent(
            """
            Hujan selama tiga hari memicu longsor besar di jalur pegunungan. Satu desa terpencil
            terisolasi total karena jembatan putus, sinyal lemah, dan jalan tertutup material.
            Di desa itu ada ibu hamil dengan kontraksi, anak demam tinggi, dan lansia yang tidak
            punya stok obat hipertensi. Sebagian warga mendengar rumor bahwa helikopter pasti
            datang malam ini, lalu menolak berjalan ke titik kumpul yang lebih aman.

            Target simulasi:
            - peta prioritas medis dan evakuasi manual bila heli belum bisa masuk
            - klasifikasi laporan warga yang tidak lengkap dan campur bahasa Sunda
            - putuskan kebutuhan alat berat, tandu, obat, dan logistik minimum 72 jam
            - uji komunikasi yang tetap tenang walau akses sangat terbatas
            """
        ).strip(),
        "priority_actions": [
            "Identifikasi korban yang harus dievakuasi manual lebih dulu bila heli belum tersedia.",
            "Larikan warga dari tebing aktif ke titik kumpul alternatif yang stabil.",
            "Kumpulkan data obat rutin, air minum, makanan bayi, dan alat komunikasi desa.",
            "Gunakan relay informasi melalui radio, motor trail, atau pos tetangga terdekat.",
            "Jangan menjanjikan heli sebelum ada konfirmasi cuaca dan akses resmi.",
        ],
        "misinformation_risks": [
            "rumor heli pasti datang malam ini",
            "pesan bahwa longsor susulan mustahil terjadi setelah hujan berhenti sebentar",
            "klaim jalan belakang desa aman padahal belum dicek",
        ],
        "vulnerable_groups": ["ibu hamil", "anak demam tinggi", "lansia dengan hipertensi"],
        "coordination_points": [
            "BPBD untuk alat berat dan jalur akses",
            "Dinkes untuk prioritas obat dan rujukan ibu hamil",
            "SAR 115 untuk evakuasi manual",
            "pemerintah desa untuk pendataan rumah terisolasi",
        ],
    },
    {
        "scenario_id": "sigap-sim-005",
        "title": "Kebakaran pasar padat, ledakan LPG, dan shelter sekolah",
        "project_name": "SIGAP-005 Urban Fire Cascade",
        "difficulty": "high",
        "hazards": ["kebakaran", "ledakan LPG", "pengungsian kota"],
        "regions": ["Makassar", "Surabaya"],
        "languages": ["Indonesia"],
        "simulation_requirement": dedent(
            """
            Kebakaran bermula di pasar padat dan merembet ke permukiman rapat. Beberapa tabung LPG
            meledak, membuat radius aman berubah cepat. Warga berdesakan ke sekolah yang dijadikan
            shelter darurat. Ada anak hilang, pedagang luka bakar, dan isu bahwa gudang bahan bakar
            di dekat lokasi juga akan meledak. Jalan sempit membuat mobil damkar sulit masuk.

            Target simulasi:
            - tentukan prioritas penyelamatan jiwa, pemisahan radius bahaya, dan triase luka bakar
            - uji informasi publik singkat yang mencegah warga mendekat untuk menonton
            - kelola pendataan keluarga terpisah dan kebutuhan shelter urban
            - paksa agen membedakan fakta ledakan aktual dan rumor ledakan berikutnya
            """
        ).strip(),
        "priority_actions": [
            "Kosongkan radius bahaya dari ledakan susulan dan bangunan yang sudah terbakar sebagian.",
            "Triase cepat luka bakar, sesak asap, dan anak yang terpisah dari keluarga.",
            "Buka jalur damkar dan ambulans dengan pengosongan kendaraan liar.",
            "Bangun meja reunifikasi keluarga di shelter sekolah.",
            "Sebarkan informasi satu pintu soal radius aman dan lokasi pengungsian resmi.",
        ],
        "misinformation_risks": [
            "rumor gudang bahan bakar akan meledak tanpa konfirmasi damkar",
            "video kebakaran kota lain diklaim lokasi yang sama",
            "broadcast bahwa sekolah shelter penuh padahal masih menerima korban",
        ],
        "vulnerable_groups": ["anak terpisah", "korban luka bakar", "pedagang lansia"],
        "coordination_points": [
            "damkar untuk radius aman dan sumber api",
            "RS untuk luka bakar dan inhalasi asap",
            "Dinsos untuk pengungsian kota",
            "polisi/linmas untuk buka akses jalan sempit",
        ],
    },
    {
        "scenario_id": "sigap-sim-006",
        "title": "Kebocoran bahan kimia pabrik dekat sekolah dan sungai",
        "project_name": "SIGAP-006 Chemical Leak Near School",
        "difficulty": "extreme",
        "hazards": ["kebocoran bahan kimia", "kontaminasi udara", "kontaminasi air"],
        "regions": ["Cilegon", "Gresik"],
        "languages": ["Indonesia"],
        "simulation_requirement": dedent(
            """
            Tangki bahan kimia industri bocor di kawasan dekat sekolah, permukiman buruh, dan aliran
            sungai kecil yang menjadi sumber air warga. Gejala awal: mata perih, batuk, muntah, dan
            kepanikan orang tua yang menjemput anak. Sebagian warga menyiram area dengan air sembarangan,
            sebagian lagi menyebar rumor bahwa semua sumur sudah beracun permanen. Belum jelas apakah
            kebocoran bersifat mudah terbakar, beracun lewat inhalasi, atau keduanya.

            Target simulasi:
            - uji protokol awal zona aman, evakuasi sekolah, dan perlindungan petugas
            - tuntut SIGAP tidak memberi saran medis atau kimia yang mengada-ada
            - pisahkan gejala terpapar, kebutuhan dekontaminasi, dan hoaks total contamination
            - paksa agen meminta jenis bahan, arah angin, dan radius paparan sebelum memberi instruksi
            """
        ).strip(),
        "priority_actions": [
            "Jauhkan warga dari plume atau arah angin paparan dan evakuasi sekolah secara teratur.",
            "Minta identifikasi bahan, SDS, dan radius bahaya resmi sebelum instruksi lanjutan.",
            "Larangan menyentuh cairan bocor atau menyiram tanpa arahan hazmat.",
            "Pisahkan jalur korban terpapar ringan, berat, dan warga sehat yang hanya panik.",
            "Lindungi sumber air sementara dan hentikan penggunaan air yang dicurigai sampai diuji.",
        ],
        "misinformation_risks": [
            "semua sumur dipastikan beracun permanen tanpa hasil uji",
            "air biasa bisa dipakai bebas untuk menetralkan semua bahan kimia",
            "warga menyebar rumor ledakan besar padahal belum ada konfirmasi sifat bahan",
        ],
        "vulnerable_groups": ["anak sekolah", "petugas tanpa APD", "warga dengan penyakit paru"],
        "coordination_points": [
            "damkar hazmat atau unit teknis industri",
            "RS dan PSC untuk gejala paparan",
            "BPBD untuk evakuasi radius bahaya",
            "dinas lingkungan untuk pengujian air",
        ],
    },
    {
        "scenario_id": "sigap-sim-007",
        "title": "Siklon tropis, gelombang tinggi, dan feri terdampar",
        "project_name": "SIGAP-007 Cyclone and Ferry Isolation",
        "difficulty": "high",
        "hazards": ["cuaca ekstrem", "gelombang tinggi", "isolasi laut"],
        "regions": ["NTT", "Maluku"],
        "languages": ["Indonesia", "Indonesia Timur"],
        "simulation_requirement": dedent(
            """
            Siklon tropis memicu gelombang tinggi, hujan deras, dan putusnya pelayaran antarpulau.
            Sebuah feri penuh penumpang terpaksa berlindung di teluk sempit, sementara warga pulau
            kecil kehabisan BBM, air bersih, dan sinyal. Ada penumpang hamil tua, anak demam, serta
            rumor bahwa kapal bisa berangkat sendiri malam ini walau syahbandar belum membuka jalur.

            Target simulasi:
            - uji keputusan stay-put vs evakuasi pada kondisi laut buruk
            - olah laporan lapangan dengan ragam Indonesia Timur
            - prioritaskan kebutuhan medis ringan, air, dan komunikasi penumpang feri
            - paksa agen membedakan info syahbandar resmi dan rumor awak tidak resmi
            """
        ).strip(),
        "priority_actions": [
            "Pegang keputusan syahbandar dan cuaca resmi, jangan dorong pelayaran spekulatif.",
            "Hitung stok air minum, obat dasar, dan makanan untuk penumpang feri serta pulau kecil.",
            "Tentukan siapa yang butuh evakuasi medis prioritas bila cuaca membuka celah singkat.",
            "Bangun informasi berkala agar kepanikan penumpang tidak meledak.",
            "Pisahkan kebutuhan pulau kecil yang kritis dari kebutuhan feri agar logistik tepat sasaran.",
        ],
        "misinformation_risks": [
            "rumor kapal bisa jalan malam ini tanpa izin syahbandar",
            "cuaca palsu dari akun nonresmi",
            "klaim stok air aman padahal data distribusi belum dicek",
        ],
        "vulnerable_groups": ["ibu hamil tua", "anak demam", "penumpang dengan penyakit kronis"],
        "coordination_points": [
            "BMKG untuk cuaca laut",
            "syahbandar dan operator feri",
            "BPBD kabupaten kepulauan",
            "puskesmas terapung atau layanan medis pelabuhan",
        ],
    },
    {
        "scenario_id": "sigap-sim-008",
        "title": "Karhutla, asap lintas kabupaten, dan sekolah lumpuh",
        "project_name": "SIGAP-008 Haze and Wildfire Health Stress",
        "difficulty": "high",
        "hazards": ["karhutla", "asap", "krisis kesehatan"],
        "regions": ["Riau", "Jambi", "Kalimantan Barat"],
        "languages": ["Indonesia", "Melayu lokal"],
        "simulation_requirement": dedent(
            """
            Kebakaran lahan dan gambut menyebar cepat saat kemarau. Asap tebal menutup sekolah,
            pasar, dan jalur penerbangan lokal. Warga mulai sesak, anak sekolah belajar dari rumah,
            dan masker berkualitas habis. Sebagian tokoh lokal menilai asap hanya "kabut biasa",
            sedangkan warga lain panik membeli obat tanpa resep. Relawan bingung kapan fokus pada
            pemadaman, kapan fokus pada perlindungan kelompok rentan.

            Target simulasi:
            - uji prioritas kesehatan publik, distribusi masker, dan pembatasan aktivitas luar ruang
            - bedakan kabut biasa, asap karhutla, dan rumor kebocoran gas
            - peta kebutuhan oksigen, ruang aman berfilter, dan perlindungan sekolah
            - paksa agen memberi instruksi singkat yang tidak menyepelekan ISPU dan gejala sesak
            """
        ).strip(),
        "priority_actions": [
            "Batasi aktivitas luar ruang saat kualitas udara buruk dan prioritaskan kelompok rentan.",
            "Distribusikan masker yang layak dan ruang aman berfilter untuk bayi serta lansia.",
            "Pantau gejala sesak, dehidrasi, dan eksaserbasi asma setiap hari.",
            "Jaga jalur evakuasi medis dan transport pasien saat jarak pandang turun.",
            "Gunakan istilah sederhana agar warga tidak menganggap asap sebagai kabut biasa.",
        ],
        "misinformation_risks": [
            "asap dianggap aman karena hanya kabut musiman",
            "warga membeli obat sembarangan tanpa penilaian medis",
            "rumor semua sekolah pasti aman dibuka besok tanpa melihat kualitas udara",
        ],
        "vulnerable_groups": ["bayi", "lansia", "warga dengan asma"],
        "coordination_points": [
            "BMKG dan otoritas kualitas udara",
            "Dinkes untuk pemantauan ISPA",
            "BPBD dan Manggala Agni untuk hotspot",
            "sekolah dan dinas pendidikan untuk pembelajaran darurat",
        ],
    },
    {
        "scenario_id": "sigap-sim-009",
        "title": "Gempa kota besar, gedung retak, dan jaringan komuter lumpuh",
        "project_name": "SIGAP-009 Urban Quake Commuter Collapse",
        "difficulty": "high",
        "hazards": ["gempa kota", "gedung retak", "transport lumpuh"],
        "regions": ["Bandung", "Jakarta", "Bekasi"],
        "languages": ["Indonesia", "Sunda", "Betawi"],
        "simulation_requirement": dedent(
            """
            Gempa kuat merusak gedung perkantoran, apartemen, halte, dan stasiun komuter saat jam
            pulang kerja. Ribuan pekerja terjebak macet, lift berhenti, dan sinyal seluler padat.
            Sebagian gedung belum roboh tetapi retak berat. Media sosial dipenuhi unggahan lokasi
            "aman" yang tidak jelas, sementara keluarga saling mencari lewat tagar darurat.

            Target simulasi:
            - prioritaskan keputusan masuk/keluar gedung, titik kumpul, dan pengelolaan kerumunan
            - bangun alur reunifikasi keluarga dan pekerja komuter
            - bedakan laporan gedung retak, bangunan runtuh, dan titik medis yang masih aktif
            - paksa agen mengelola panic buying dan kemacetan yang menghambat ambulans
            """
        ).strip(),
        "priority_actions": [
            "Kosongkan gedung retak berat dan larang warga masuk mengambil barang.",
            "Tetapkan titik kumpul terbuka di luar jalur kaca, billboard, dan jembatan layang.",
            "Buka jalur ambulans di tengah kemacetan dan atur titik medis darurat.",
            "Kumpulkan daftar orang hilang dan titik reunifikasi keluarga.",
            "Saring unggahan lokasi aman yang belum diverifikasi oleh petugas lapangan.",
        ],
        "misinformation_risks": [
            "unggahan lokasi aman tanpa verifikasi struktur",
            "rumor jembatan utama pasti ambruk padahal belum dicek",
            "broadcast bahwa semua transport publik dihentikan total tanpa info operator resmi",
        ],
        "vulnerable_groups": ["pekerja terjebak lift", "anak yang pulang sekolah", "pasien di gedung tinggi"],
        "coordination_points": [
            "BMKG untuk info gempa susulan",
            "BPBD dan damkar untuk asesmen bangunan",
            "operator KRL/MRT/LRT untuk status komuter",
            "RS dan PSC untuk titik triase",
        ],
    },
    {
        "scenario_id": "sigap-sim-010",
        "title": "Wabah diare dan demam di pengungsian pascabanjir",
        "project_name": "SIGAP-010 Shelter Outbreak After Flood",
        "difficulty": "high",
        "hazards": ["wabah pengungsian", "banjir", "sanitasi buruk"],
        "regions": ["Kalimantan Selatan", "Aceh"],
        "languages": ["Indonesia", "Banjar", "Aceh mixed"],
        "simulation_requirement": dedent(
            """
            Seminggu setelah banjir besar, pengungsian padat mulai mengalami lonjakan diare, demam,
            dan muntah. Toilet darurat tidak cukup, air bersih terbatas, dan dapur umum mulai
            kehabisan sabun. Rumor vaksin atau oralit beracun beredar. Warga bingung apakah harus
            memisahkan balita sakit, apakah air sumur masih boleh dipakai, dan kapan perlu rujuk
            ke rumah sakit.

            Target simulasi:
            - uji protokol kesehatan dasar di shelter tanpa membuat panik
            - prioritaskan WASH, oralit, pemisahan gejala berat, dan pelaporan kasus
            - bantah rumor racun oralit atau vaksin palsu
            - paksa agen menyeimbangkan bahasa medis sederhana dan instruksi praktis
            """
        ).strip(),
        "priority_actions": [
            "Pisahkan kasus berat, dehidrasi, dan demam tinggi dari area umum shelter.",
            "Amankan air bersih, sabun, toilet, dan oralit sebagai prioritas logistik.",
            "Pantau balita, lansia, dan ibu hamil untuk tanda bahaya lebih dini.",
            "Gunakan pesan sederhana soal cuci tangan, air matang, dan rujukan kasus berat.",
            "Laporkan tren gejala harian agar wabah shelter tidak terlambat terdeteksi.",
        ],
        "misinformation_risks": [
            "oralit dianggap racun",
            "vaksin atau obat dasar dianggap penyebab diare",
            "sumur dinyatakan aman total tanpa uji kualitas air",
        ],
        "vulnerable_groups": ["balita", "ibu hamil", "lansia"],
        "coordination_points": [
            "Dinkes dan puskesmas untuk surveilans kasus",
            "BPBD/Dinsos untuk shelter dan WASH",
            "RS rujukan untuk dehidrasi berat",
            "relawan sanitasi untuk toilet dan air bersih",
        ],
    },
    {
        "scenario_id": "sigap-sim-011",
        "title": "Lahar dingin setelah erupsi dan hujan deras malam hari",
        "project_name": "SIGAP-011 Cold Lahar Night Emergency",
        "difficulty": "high",
        "hazards": ["lahar dingin", "banjir material", "evakuasi malam"],
        "regions": ["Magelang", "Sleman"],
        "languages": ["Indonesia", "Jawa"],
        "simulation_requirement": dedent(
            """
            Setelah erupsi beberapa hari lalu, hujan sangat deras pada malam hari memicu lahar dingin
            di sungai-sungai berhulu gunung. Jembatan desa terancam, warga bantaran sungai mendengar
            suara gemuruh, dan listrik padam di sejumlah dusun. Ada peternak yang ingin tetap menjaga
            kandang dekat sungai, sementara warga lain mengira ancaman hanya abu, bukan banjir material.

            Target simulasi:
            - uji keputusan evakuasi malam dengan penerangan terbatas
            - pisahkan bahaya erupsi aktif dan bahaya lahar dingin
            - prioritaskan jembatan, bantaran sungai, kandang ternak, dan jalur keluar dusun
            - paksa agen memberi instruksi sangat singkat yang mudah dipakai pengeras suara desa
            """
        ).strip(),
        "priority_actions": [
            "Kosongkan bantaran sungai dan area jembatan sebelum material turun lebih besar.",
            "Gunakan pengeras suara desa untuk instruksi pendek, jelas, dan berulang.",
            "Larangan berjaga di kandang dekat alur lahar bila keselamatan jiwa terancam.",
            "Tentukan jalur keluar dusun yang tidak memotong sungai atau jembatan rapuh.",
            "Pisahkan update lahar dingin dari update erupsi agar warga tidak salah fokus.",
        ],
        "misinformation_risks": [
            "warga mengira ancaman hanya abu vulkanik",
            "rumor sungai tertentu pasti aman tanpa patroli visual",
            "pesan bahwa jembatan kuat karena siang tadi masih dilalui",
        ],
        "vulnerable_groups": ["warga bantaran sungai", "anak kecil", "peternak lansia"],
        "coordination_points": [
            "BPBD kabupaten dan relawan sungai",
            "pemerintah desa untuk pengeras suara",
            "PVMBG untuk konteks erupsi",
            "SAR 115 untuk evakuasi malam",
        ],
    },
    {
        "scenario_id": "sigap-sim-012",
        "title": "Multi-hazard libur panjang: rob, angin kencang, dan pelabuhan padat",
        "project_name": "SIGAP-012 Holiday Crowd Multi-Hazard",
        "difficulty": "extreme",
        "hazards": ["rob", "angin kencang", "massa padat", "gangguan pelabuhan"],
        "regions": ["Semarang", "Surabaya", "Bali"],
        "languages": ["Indonesia", "Jawa", "Bali"],
        "simulation_requirement": dedent(
            """
            Pada libur panjang nasional, pelabuhan dan terminal sedang sangat padat ketika rob,
            angin kencang, dan hujan ekstrem terjadi bersamaan. Penumpang panik karena jadwal kapal
            berubah, sebagian memaksa antre, dan anak-anak mulai terpisah dari keluarga. Di media
            sosial muncul kabar bahwa semua kapal dibatalkan selama tiga hari, padahal operator belum
            memberi pengumuman final. Pedagang kaki lima dan sopir informal ikut menyebarkan kabar
            yang simpang siur.

            Target simulasi:
            - uji manajemen kerumunan, informasi antrean, dan prioritas kelompok rentan
            - paksa agen menyeimbangkan keselamatan pelabuhan dengan kebutuhan penumpang
            - bangun alur informasi resmi yang melawan rumor pembatalan total
            - nilai kebutuhan shelter transit, air minum, charger, dan titik reunifikasi
            """
        ).strip(),
        "priority_actions": [
            "Stabilkan kerumunan dengan informasi antrean yang rutin dan jelas.",
            "Prioritaskan anak, lansia, ibu hamil, dan penumpang sakit untuk shelter transit.",
            "Pisahkan status kapal resmi dari rumor pedagang atau sopir informal.",
            "Siapkan titik reunifikasi keluarga, air minum, pengisian daya, dan toilet.",
            "Tutup akses ke dermaga rawan saat angin dan gelombang melampaui batas aman.",
        ],
        "misinformation_risks": [
            "semua kapal dibatalkan tiga hari tanpa pengumuman operator",
            "jalur antrean VIP palsu yang bikin kerumunan pindah liar",
            "rumor rob akan masuk terminal utama dalam hitungan menit tanpa data pasang",
        ],
        "vulnerable_groups": ["anak terpisah", "ibu hamil", "penumpang sakit"],
        "coordination_points": [
            "BMKG untuk cuaca dan pasang",
            "operator pelabuhan/terminal",
            "BPBD dan Dinsos untuk shelter transit",
            "petugas kesehatan dan PSC di simpul transportasi",
        ],
    },
]


def dedent_clean(text: str) -> str:
    return dedent(text).strip()


def render_priority_answer(scenario: dict) -> str:
    actions = "\n".join(f"{idx}. {item}" for idx, item in enumerate(scenario["priority_actions"], start=1))
    vulnerable = ", ".join(scenario["vulnerable_groups"])
    coordination = ", ".join(scenario["coordination_points"])
    return (
        f"Prioritas SIGAP untuk 0-6 jam pertama pada skenario {scenario['title']} adalah:\n"
        f"{actions}\n\n"
        f"Kelompok rentan yang wajib dipantau lebih dulu: {vulnerable}.\n"
        f"Koordinasi utama: {coordination}.\n\n"
        "SIGAP tidak boleh mengarang data real-time. Semua instruksi lapangan harus menekankan "
        "verifikasi ke BMKG/PVMBG, BNPB/BPBD, SAR 115, Ambulans 118, atau 112 sesuai jenis ancaman."
    )


def render_misinfo_answer(scenario: dict) -> str:
    risks = "\n".join(f"- {item}" for item in scenario["misinformation_risks"])
    return (
        f"Dalam skenario {scenario['title']}, SIGAP harus menyaring laporan dengan urutan berikut:\n"
        "1. Tandai mana laporan saksi langsung, mana broadcast ulang, mana klaim tanpa sumber.\n"
        "2. Cocokkan waktu, lokasi, arah ancaman, dan status resmi sebelum memberi instruksi.\n"
        "3. Ubah laporan campur bahasa menjadi ringkasan operasional: lokasi, korban, akses, kebutuhan, dan risiko.\n"
        "4. Jika ada hoaks, bantah dengan singkat dan arahkan warga ke kanal resmi.\n\n"
        f"Risiko hoaks utama yang harus diwaspadai:\n{risks}\n\n"
        "Output SIGAP harus berupa ringkasan situasi yang bersih, daftar prioritas, dan catatan apa yang "
        "masih belum terverifikasi."
    )


def render_coordination_answer(scenario: dict) -> str:
    coordination = "\n".join(f"- {item}" for item in scenario["coordination_points"])
    data_min = [
        "lokasi yang presisi atau landmark terdekat",
        "jenis ancaman utama dan apakah masih aktif",
        "jumlah korban, kelompok rentan, dan kondisi medis singkat",
        "status akses jalan, listrik, komunikasi, serta sumber air",
        "kebutuhan logistik 6-24 jam pertama",
    ]
    data_min_lines = "\n".join(f"- {item}" for item in data_min)
    return (
        f"Koordinasi lintas lembaga untuk skenario {scenario['title']} harus dibangun sebagai berikut:\n"
        f"{coordination}\n\n"
        "Data minimal yang harus dikumpulkan SIGAP sebelum eskalasi keputusan:\n"
        f"{data_min_lines}\n\n"
        "Setelah data minimal terkumpul, SIGAP perlu mengeluarkan tiga hal: "
        "prioritas tindakan, daftar kebutuhan logistik kritis, dan keputusan apa yang harus "
        "diambil operator lapangan dalam 1-3 jam ke depan."
    )


def build_dataset_records(scenarios: list[dict]) -> list[dict]:
    records: list[dict] = []
    for scenario in scenarios:
        records.extend(
            [
                {
                    "q": f"Dalam skenario {scenario['title']}, apa prioritas SIGAP pada 0-6 jam pertama?",
                    "a": render_priority_answer(scenario),
                },
                {
                    "q": f"Jika laporan lapangan pada skenario {scenario['title']} campur bahasa dan banyak hoaks, bagaimana SIGAP harus menyaring informasi?",
                    "a": render_misinfo_answer(scenario),
                },
                {
                    "q": f"Untuk skenario {scenario['title']}, bagaimana koordinasi lintas lembaga dan data minimal yang harus dikumpulkan SIGAP?",
                    "a": render_coordination_answer(scenario),
                },
            ]
        )
    return records


def normalize_scenarios() -> list[dict]:
    normalized = []
    for item in SCENARIOS:
        cleaned = dict(item)
        cleaned["simulation_requirement"] = dedent_clean(cleaned["simulation_requirement"])
        cleaned["recommended_seed_file"] = str(SEED_PATH)
        normalized.append(cleaned)
    return normalized


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    scenarios = normalize_scenarios()
    dataset_records = build_dataset_records(scenarios)

    write_json(PACK_PATH, scenarios)
    SEED_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEED_PATH.write_text(SEED_TEXT, encoding="utf-8")
    write_json(DATASET_PATH, dataset_records)
    write_json(
        MANIFEST_PATH,
        {
            "scenario_pack": str(PACK_PATH),
            "seed_file": str(SEED_PATH),
            "dataset_file": str(DATASET_PATH),
            "scenario_count": len(scenarios),
            "dataset_record_count": len(dataset_records),
        },
    )

    print(f"Wrote scenario pack: {PACK_PATH}")
    print(f"Wrote seed file: {SEED_PATH}")
    print(f"Wrote dataset file: {DATASET_PATH}")
    print(f"Scenarios: {len(scenarios)} | Dataset records: {len(dataset_records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
