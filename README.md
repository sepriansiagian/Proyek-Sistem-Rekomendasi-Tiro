<<<<<<< HEAD
# Proyek Sistem Rekomendasi Tiro

## W6 — Baseline dan sanity check

### Persiapan

Gunakan Python 3.10+ dan install dependensi berikut dari root repository:

```powershell
python -m pip install numpy pandas scipy
```

### Jalankan empat baseline

```powershell
python run_baselines.py --output-dir outputs/w6_final
```

Skrip menjalankan Popularity, Truncated SVD MF, TF-IDF CBF, dan popularity-aware MF reranking. Hasilnya meliputi konfigurasi, metrik, NDCG per pengguna, dan contoh Top-10. Opsi `--output-dir` menyimpan run di folder baru sehingga hasil lama tidak tertimpa.

### Jalankan sanity check

Sanity check membaca baseline tersimpan di `outputs/w6/metrics.json`, membangun ulang ranking, lalu memeriksa jumlah rekomendasi, keunikan item, filtering item yang sudah dilihat, dan kecocokan metrik.

```powershell
python run_sanity_audit.py --output-dir outputs/w6_sanity_final
```

Gunakan baseline yang tersimpan di `outputs/w6/` sebagai input audit. Jangan menghapus atau mengganti file baseline tersebut sebelum audit selesai.

### Data dan aturan evaluasi

- `data/interactions_train.csv`: data untuk fitting model dan menghitung popularitas.
- `data/interactions_validation.csv`: target evaluasi; tidak dipakai untuk fitting.
- `data/candidate_items_warm.csv`: candidate set yang sama untuk seluruh baseline.
- `data/items.csv`: metadata produk untuk TF-IDF CBF.
- Relevansi: rating validation ≥4; cutoff Top-10.
- Item yang sudah dilihat pengguna pada training dikeluarkan dari rekomendasi.
- Long-tail: jumlah interaksi training item ≤5.

### Output utama

- `outputs/w6/`: konfigurasi, metrik, NDCG per pengguna, contoh Top-10, dan sanity check yang sudah tersimpan.
- `outputs/w6_sanity_rerun/`: hasil sanity check run tambahan.
- `outputs/w6_sanity_final/`: folder contoh untuk hasil sanity check baru.

Validation telah digunakan untuk evaluasi; data yang tersedia tidak mencakup test set independen.
=======
# Proyek-Sistem-Rekomendasi-Tiro
>>>>>>> origin/main
