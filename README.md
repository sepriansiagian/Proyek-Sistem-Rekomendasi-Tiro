# Proyek Sistem Rekomendasi Tiro

Kelompok 3 — Track C: Popularity Bias & Long Tail. Dataset: Amazon Reviews 2023 – Musical Instruments (5-core).

Gunakan Python 3.10+ dan install dependensi dari root repository:

```powershell
python -m pip install numpy pandas scipy
```

## CP3 — Desain eksperimen

| File | Isi |
|---|---|
| `cp3_protocol.py` | Protokol warm-start bersama: split, evaluable users, candidate set, metrik (Recall@10, NDCG@10, metrik Track C), dan guard leakage. Test split terkunci. |
| `cp3_models.py` | Popularity, Truncated SVD MF, TF-IDF CBF, dan MF popularity-aware reranking yang bisa diparametrisasi. |
| `configs/cp3_search_space.json` | Experiment matrix E1–E4, search space, seed, dan aturan model selection. |
| `run_experiments.py` | Menjalankan grid pada validation, mencatat setiap run, dan menerapkan aturan selection. |
| `check_w6_reproduction.py` | Memastikan pipeline CP3 mereproduksi metrik W6, serta menambahkan Recall@10. |
| `tests/test_cp3_protocol.py` | Sanity test metrik, ranking, dan guard leakage. |
| `reports/cp3/` | Draf teks Overleaf (Experimental Setup) dan progress report CP3. |

```powershell
python cp3_protocol.py                                 # laporan protokol + cek leakage -> outputs/cp3/protocol_report.json
python -m unittest discover -s tests -v                # sanity test metrik
python check_w6_reproduction.py                        # reproduksi W6 + Recall@10 -> outputs/cp3/w6_reproduction.json
python run_experiments.py --dry-run                    # daftar run id dalam experiment matrix
python run_experiments.py --experiments E1 E2 E3       # jalankan baseline (W10-W12)
python run_experiments.py --experiments E4             # Track C, memakai factors MF terpilih dari E2
python run_experiments.py --select                     # terapkan aturan selection -> experiments/selection.json
```

Setiap run menulis `experiments/runs/<run_id>/` (config, metrik, metrik per user) dan menambahkan satu baris ke `experiments/experiment_log.csv`. Baris tersebut memuat commit git, flag *dirty*, sidik data (SHA-256), seed, dan versi config. Commit perubahan kode sebelum menjalankan run yang akan dilaporkan.

Penamaan run: `<Eksperimen>_<model>_<parameter>_s<seed>`, misalnya `E2_mf_f64_s42` atau `E4_mf_rerank_f64_a10_b12_s42`.

## W10 — Evaluasi validation dan uji awal Track C

Hanya split validation yang dipakai; test terkunci. `BATCH = 128` di `run_experiments.py` (E4 menghitung 25 matriks skor per batch). Commit kode lebih dulu agar `git_dirty` di log bersih.

```powershell
python cp3_protocol.py                                      # cek leakage (temporal, overlap=0, 11.850 evaluable users)
python -m unittest discover -s tests -v                     # test metrik + validasi manual 3 user
python run_experiments.py --experiments E1 E2 E3            # baseline (10 run)
python run_experiments.py --experiments E4                  # Track C (25 run)
python run_experiments.py --select                          # -> experiments/selection.json
python make_w10_report.py                                   # -> reports/w10/hasil_validation.{csv,md}
```

| File | Isi |
|---|---|
| `make_w10_report.py` | Membaca `experiments/experiment_log.csv`; membuat tabel Recall@10, NDCG@10, coverage, long-tail exposure/recall. |
| `reports/w10/hasil_validation.md` / `.csv` | Tabel E1, E2, E3, dan E4 terpilih + grid alpha x beta. |
| `reports/w10/analisis_awal.md` | Analisis awal hasil, anomali, keterbatasan, dan insight W11–W12. |
| `reports/w10/validasi_manual_metrik.md` | Perhitungan manual metrik untuk 3 user (test: `tests/test_manual_metric_validation.py`). |

## W6 — Baseline dan sanity check

```powershell
python run_baselines.py --output-dir outputs/w6_final
python run_sanity_audit.py --output-dir outputs/w6_sanity_final
```

`run_baselines.py` menjalankan Popularity, Truncated SVD MF, TF-IDF CBF, dan popularity-aware MF reranking. Sanity check membaca `outputs/w6/metrics.json`, membangun ulang ranking, lalu memeriksa jumlah rekomendasi, keunikan item, filtering item yang sudah dilihat, dan kecocokan metrik. Jangan menghapus atau mengganti isi `outputs/w6/`.

## Data dan aturan evaluasi

- `data/interactions_train.csv`: satu-satunya sumber untuk fitting model dan menghitung popularitas.
- `data/interactions_validation.csv`: target evaluasi dan model selection; tidak dipakai untuk fitting.
- `data/candidate_items_warm.csv`: candidate universe yang sama untuk seluruh model.
- `data/items.csv`: metadata produk (`title`, `content_text_basic`) untuk TF-IDF CBF.
- Relevansi: rating validation ≥4. Cutoff Top-10. Item train milik user dikeluarkan dari kandidat.
- Long-tail: item dengan ≤5 interaksi training.
- Test set tidak ada di repository dan tidak digunakan sebelum evaluasi final W13.
