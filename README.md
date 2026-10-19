# Proyek Sistem Rekomendasi Tiro

## W6 — Bias Popularitas & Long Tail

### Menjalankan baseline

Python 3.10+ dan paket `numpy`, `pandas`, serta `scipy` diperlukan. Dari root repository:

```powershell
python -m pip install numpy pandas scipy
python run_baselines.py --output-dir outputs/w6_rerun_YYYYMMDD
```

Gunakan nama folder keluaran baru untuk mempertahankan hasil terdahulu. Empat baseline yang dijalankan adalah popularity, Truncated SVD MF, TF-IDF CBF, dan MF popularity-aware reranking. Skrip menghasilkan `config.json`, `metrics.json`, `per_user_ndcg.csv`, dan `top_n_examples.csv` di folder yang dipilih.

Eksperimen tambahan dapat dijalankan terpisah:

```powershell
python run_mf_rerank_sweep.py --output-dir outputs/w6_reranking_rerun_YYYYMMDD
python run_long_tail_relevance_sweep.py --output-dir outputs/w6_long_tail_rerun_YYYYMMDD
python run_observed_mf_comparator.py --output-dir outputs/w6_observed_mf_rerun_YYYYMMDD
python run_sanity_audit.py --output-dir outputs/w6_sanity_rerun_YYYYMMDD
```

Observed-only MF menggunakan L-BFGS-B; fit 1.000 iterasi yang tercatat belum konvergen. Jangan menafsirkan hasilnya sebagai baseline terpilih.

### Data dan protokol evaluasi

- `data/interactions_train.csv`: satu-satunya sumber interaksi untuk melatih model dan menghitung popularitas.
- `data/interactions_validation.csv`: target evaluasi/pemilihan parameter; tidak digunakan untuk fitting model.
- `data/candidate_items_warm.csv`: candidate warm yang sama untuk semua model, sebanyak 22.753 item.
- `data/items.csv`: metadata produk untuk TF-IDF.
- Relevansi: rating validation ≥ 4; Top-10; item yang telah dilihat pengguna pada training dikeluarkan.
- Long-tail: jumlah interaksi training item ≤ 5.

Validation sudah dipakai memilih konfigurasi. Tidak ada test set independen yang ditemukan; hasil sweep bukan evaluasi final independen.

### Lokasi artefak yang tercatat

- `outputs/w6/`: baseline utama, konfigurasi, metrik, NDCG per pengguna, contoh Top-10, dan sanity check.
- `outputs/w6_followup/`: sweep reranking awal, perbandingan MF, serta fit observed-only MF 25/150/500/1.000 iterasi dan faktor tersimpan.
- `outputs/w6_long_tail_relevance/`: sweep gated reranking, protokol, Recall@10 long-tail, dan tabel gabungan.
- `docs/`: tabel ringkasan, draf Method Overleaf, ringkasan Progress Report W6, catatan AI, kontribusi, review, dan draf PR. `method_w6_overleaf.tex` dapat diunggah ke Overleaf.
- `run_sanity_audit.py`: menghitung ulang ranking/metric checks dan menyimpan audit serta contoh Top-10 ke folder baru.

Kode keluaran default masih menunjuk folder historis. Untuk rerun, gunakan `--output-dir` baru seperti contoh di atas agar artefak lama tidak tertimpa.
