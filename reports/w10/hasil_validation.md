# Hasil Validation W10

Sumber: `experiments/experiment_log.csv` (35 run, seed 42, split validation, 11850 evaluable users, git_dirty semua False). Test split tidak dipakai.

Kolom *Terpilih* menandai konfigurasi yang dipilih aturan seleksi pre-registered (E2/E3: NDCG@10 tertinggi; E4: long-tail exposure tertinggi dengan NDCG@10 >= 0,95 x NDCG@10 MF terpilih).

## Tabel utama

| Run | Model | Parameter | Recall@10 | NDCG@10 | Coverage@10 | LT exposure@10 | LT recall@10 | Terpilih |
|---|---|---|---|---|---|---|---|---|
| E1_popularity_s42 | popularity | - | 0.01336 | 0.00922 | 0.00079 | 0.00000 | 0.00000 | ya |
| E2_mf_f32_s42 | mf | factors=32 | 0.01308 | 0.00783 | 0.00954 | 0.00000 | 0.00000 | ya |
| E2_mf_f64_s42 | mf | factors=64 | 0.01235 | 0.00780 | 0.01863 | 0.00000 | 0.00000 |  |
| E2_mf_f128_s42 | mf | factors=128 | 0.01102 | 0.00685 | 0.03468 | 0.00000 | 0.00000 |  |
| E3_tfidf_mf5000_ng1_s42 | tfidf | max_features=5000, ngram_max=1 | 0.01158 | 0.00711 | 0.71841 | 0.21105 | 0.00935 |  |
| E3_tfidf_mf5000_ng2_s42 | tfidf | max_features=5000, ngram_max=2 | 0.01139 | 0.00701 | 0.71889 | 0.20964 | 0.01000 |  |
| E3_tfidf_mf10000_ng1_s42 | tfidf | max_features=10000, ngram_max=1 | 0.01168 | 0.00720 | 0.72694 | 0.20534 | 0.01109 |  |
| E3_tfidf_mf10000_ng2_s42 | tfidf | max_features=10000, ngram_max=2 | 0.01174 | 0.00710 | 0.72294 | 0.20402 | 0.00957 |  |
| E3_tfidf_mf20000_ng1_s42 | tfidf | max_features=20000, ngram_max=1 | 0.01224 | 0.00733 | 0.72382 | 0.20638 | 0.01065 | ya |
| E3_tfidf_mf20000_ng2_s42 | tfidf | max_features=20000, ngram_max=2 | 0.01071 | 0.00679 | 0.72953 | 0.20341 | 0.01065 |  |
| E4_mf_rerank_f32_a5_b12_s42 | mf_rerank | mf_factors=32, alpha=5, beta=12 | 0.01221 | 0.00750 | 0.02391 | 0.09862 | 0.00109 | ya |

### Grid E4: NDCG@10 (baris = alpha, kolom = beta)

| alpha | beta=0 | beta=6 | beta=12 | beta=18 | beta=24 |
|---|---|---|---|---|---|
| 0 | 0.00783 | 0.00783 | 0.00773 | 0.00695 | 0.00547 |
| 1 | 0.00783 | 0.00783 | 0.00769 | 0.00678 | 0.00540 |
| 5 | 0.00779 | 0.00778 | 0.00750 | 0.00625 | 0.00525 |
| 10 | 0.00771 | 0.00766 | 0.00678 | 0.00566 | 0.00497 |
| 20 | 0.00755 | 0.00646 | 0.00548 | 0.00474 | 0.00405 |

### Grid E4: Long-tail exposure@10 (baris = alpha, kolom = beta)

| alpha | beta=0 | beta=6 | beta=12 | beta=18 | beta=24 |
|---|---|---|---|---|---|
| 0 | 0.00000 | 0.00285 | 0.02818 | 0.21745 | 0.45759 |
| 1 | 0.00000 | 0.00359 | 0.03629 | 0.24212 | 0.47779 |
| 5 | 0.00041 | 0.00778 | 0.09862 | 0.35003 | 0.55948 |
| 10 | 0.00344 | 0.02818 | 0.23838 | 0.48413 | 0.64581 |
| 20 | 0.03962 | 0.28575 | 0.52497 | 0.67306 | 0.76843 |

