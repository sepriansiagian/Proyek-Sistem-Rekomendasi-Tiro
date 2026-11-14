# Analisis Awal W10: Evaluasi Validation dan Uji Awal Track C

Semua angka berasal dari `experiments/experiment_log.csv` (35 run, seed 42, split validation, 11.850 evaluable users;
tabel lengkap di `hasil_validation.md`). Test split tidak dipakai. Karena hanya satu seed dan tanpa uji
signifikansi, selisih kecil di bawah ini diperlakukan sebagai indikasi, bukan temuan.

## 1. Perbandingan antar baseline

| Model (terpilih) | Recall@10 | NDCG@10 | Coverage@10 | LT exposure@10 | LT recall@10 |
|---|---|---|---|---|---|
| E1 Popularity | 0,01336 | 0,00922 | 0,0008 | 0,000 | 0,0000 |
| E2 MF (factors=32) | 0,01308 | 0,00783 | 0,0095 | 0,000 | 0,0000 |
| E3 TF-IDF (20000 fitur, unigram) | 0,01224 | 0,00733 | 0,7238 | 0,206 | 0,0107 |
| E4 MF rerank (alpha=5, beta=12) | 0,01221 | 0,00750 | 0,0239 | 0,099 | 0,0011 |

- **Akurasi:** Popularity tertinggi (NDCG@10 0,00922), disusul MF (0,00783), E4 (0,00750), dan TF-IDF (0,00733).
  Terkuat: Popularity. Terlemah: TF-IDF pada NDCG@10, tetapi E4 dan TF-IDF hampir sama pada Recall@10 (0,01221 vs 0,01224).
- **Exposure:** urutannya terbalik. TF-IDF paling baik (coverage 72,4%, long-tail exposure 20,6%, Gini 0,71, 49 dari
  4.599 target long-tail tertangkap). Popularity paling lemah: hanya 18 item berbeda yang direkomendasikan (coverage
  0,08%, Gini 0,9996, long-tail exposure 0). MF juga tidak pernah menampilkan item long-tail (exposure 0, 0 hit).
- Jadi tidak ada model yang unggul di kedua sumbu; ini inti trade-off Track C.

## 2. Anomali dan hal yang perlu dicek

1. **Popularity mengalahkan semua model personalisasi.** NDCG@10 Popularity 17,8% lebih tinggi daripada MF
   (0,00922 vs 0,00783). Pada Recall@10 selisihnya hanya 2,1% (0,01336 vs 0,01308), yang belum bisa dibedakan dari
   noise. Penyebabnya belum kami uji. Dugaan yang perlu dicek: data 5-core yang sangat jarang, split temporal
   (item populer di train cenderung tetap populer di validation), dan rata-rata hanya sekitar 2 target per user
   (24.707 baris relevan / 11.850 user). Nilai absolut kecil (Recall@10 sekitar 1,3%) karena 22.753 kandidat.
2. **Tren MF terhadap jumlah faktor:** NDCG@10 turun dari 0,00783 (32) ke 0,00780 (64) ke 0,00685 (128); Recall@10 turun
   monoton (0,01308 → 0,01235 → 0,01102). Coverage naik (0,95% → 1,9% → 3,5%) tetapi long-tail exposure tetap 0.
   Nilai terpilih (32) adalah **nilai terkecil di grid**, yaitu di tepi. Artinya optimum bisa berada di bawah 32
   dan tidak teramati. NDCG 32 dan 64 hampir sama (selisih 0,00003), sehingga aturan tie-break "fewer factors"
   tidak berperan, tetapi keputusan ini rapuh.
3. **E3 juga di tepi grid:** terpilih `max_features=20000, ngram_max=1` (nilai terbesar). Keenam konfigurasi berada
   dalam rentang sempit (NDCG 0,00679–0,00733), dan bigram tidak membantu (pada 20000 fitur malah turun, 0,00679 vs 0,00733).
   Perbedaannya kecil dan bisa jadi noise.
4. **Rerank E4 menggeser exposure ke sedikit item long-tail.** Pada konfigurasi terpilih long-tail exposure naik dari
   0 menjadi 9,9%, tetapi coverage hanya 2,4% (544 item) dan Gini 0,996, hampir sama dengan MF dasar. Hit long-tail
   hanya 5 (LT recall 0,0011), sekitar 10 kali lebih rendah daripada TF-IDF (49 hit, 0,0107). Reranking menaikkan
   "jumlah" tampilan item ekor, belum tentu item ekor yang relevan.

## 3. Dampak alpha dan beta pada E4 (grid 5x5, MF factors=32)

- **Beta (bonus long-tail) adalah pengungkit utama.** Dengan alpha=0, long-tail exposure naik 0,000 → 0,003 → 0,028 →
  0,217 → 0,458 untuk beta 0 → 24, sementara NDCG turun 0,00783 → 0,00783 → 0,00773 → 0,00695 → 0,00547.
  Perubahan terjadi seperti tebing antara beta 12 dan 18: exposure naik sekitar 8x, NDCG turun sekitar 10%.
- **Alpha (penalti popularitas) saja kurang efektif:** dengan beta=0, alpha=20 hanya mencapai exposure 0,040 dengan
  NDCG 0,00755. Alpha memperkuat efek beta (misalnya beta=12: exposure 0,028 pada alpha=0, 0,099 pada alpha=5,
  0,238 pada alpha=10, 0,525 pada alpha=20).
- **Trade-off tajam:** exposure tertinggi di grid (0,768 pada alpha=20, beta=24) membuat NDCG turun ke 0,00405
  (52% dari MF dasar). LT recall di seluruh grid maksimum 0,0085, masih di bawah TF-IDF (0,0107).
- **Konfigurasi terpilih** (alpha=5, beta=12) memenuhi batas NDCG ≥ 0,95 × 0,00783 = 0,00744 dengan NDCG 0,00750
  (95,8% dari MF), yaitu sangat dekat batas. Titik ini berada di tengah grid, tidak di tepi. Namun batasnya jatuh di celah
  antara beta 12 dan 18, sehingga grid terlalu kasar untuk menemukan titik yang lebih baik.

## 4. Keterbatasan

- Satu seed (42): tidak ada estimasi variansi, terutama untuk MF.
- Hanya validation; belum ada konfirmasi di test (terkunci sampai W13). Seleksi hiperparameter dan pelaporan memakai
  split yang sama sehingga ada risiko optimistis.
- Belum ada uji signifikansi atau interval kepercayaan. Banyak selisih (Popularity vs MF di Recall, antar konfigurasi
  E3, antar faktor MF) sangat kecil.
- LT recall didasarkan pada hit yang sedikit (0–49 dari 4.599 target) sehingga sangat rentan noise.
- Hanya 11.850 dari 15.748 user validation yang evaluable (warm-start); hasil tidak berlaku untuk cold-start.
- E4 mewarisi pilihan factors=32 dari E2 yang berada di tepi grid.

## 5. Insight untuk W11–W12

Usulan berikut hanya rekomendasi; search space pre-registered tidak diubah. Jika dipakai, perlu dicatat sebagai
analisis tambahan (post-hoc), bukan pengganti seleksi resmi.

1. Uji kepekaan di luar grid: MF dengan factors < 32 (mis. 8/16) dan beta di antara 12 dan 18 pada E4, dilaporkan terpisah.
2. Jalankan beberapa seed (minimal untuk MF dan E4) dan hitung bootstrap berpasangan per user memakai
   `experiments/runs/*/per_user.csv.gz` untuk menguji Popularity vs MF vs TF-IDF dan E4 vs MF.
3. Selidiki mengapa Popularity kuat: pecah hasil per kelompok aktivitas user dan per kepopuleran target.
4. Karena E4 menaikkan exposure tetapi bukan LT recall, pertimbangkan metrik/analisis yang memisahkan "tampil" dan
   "relevan", serta kombinasi sinyal konten (TF-IDF) dengan MF sebagai kandidat ekstensi.
5. Laporkan seluruh kurva trade-off alpha x beta (bukan hanya titik terpilih) untuk RQ2.
