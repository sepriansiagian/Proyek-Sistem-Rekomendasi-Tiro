# Validasi Manual Metrik Recall@10 dan NDCG@10

Tiga user evaluable dari **validation** (bukan test) dipilih dari daftar Top-10 baseline Popularity
(E1). Nilai dihitung dengan tangan lalu dicocokkan dengan `cp3_protocol.recall_at_k` /
`ndcg_at_k` dan dengan `experiments/runs/E1_popularity_s42/per_user.csv.gz`. Test otomatisnya ada
di `tests/test_manual_metric_validation.py`.

Rumus (rank 1-based, K = 10):
- Recall@10 = (jumlah target di Top-10) / (jumlah target relevan user)
- DCG = Σ 1/log2(rank+1) untuk tiap hit; IDCG = Σ_{r=1..min(10,|target|)} 1/log2(r+1); NDCG = DCG/IDCG

Top-10 Popularity (sama untuk ketiga user, karena tidak ada item training mereka di dalamnya):
1 B09857JRP2, 2 B0BCK6L7S5, 3 B0BSGM6CQ9, 4 B0BPJ4Q6FJ, 5 B0BTC9YJ2W, 6 B004XNK7AI,
7 B08R5GM6YB, 8 B0BHSG5C3G, 9 B0BPLFP5P6, 10 B08JMQR2JK.

## User A: 1 target, ada hit (`AE3BRGSQ4LSNQUORETJUS7PKBZ7Q`)
- Target: {B0BSGM6CQ9} → berada di rank 3.
- Recall@10 = 1/1 = **1,0**.
- DCG = 1/log2(4) = 0,5; IDCG = 1/log2(2) = 1 → NDCG@10 = **0,5**.

## User B: banyak target (9), ada hit (`AE23LDQTB7L76AP6E6WPBFVYL5DA`)
- Target (9): B00CGFRJ2Y, B07H4LNV2J, B07NJBLXBV, B08H75RJZV, B0B3MWSSYF, B0B6H7J2SN, B0BG95DG2H,
  B0BPJ4Q6FJ, B0BQF951BC. Hanya B0BPJ4Q6FJ ada di Top-10, di rank 4.
- Recall@10 = 1/9 = **0,111111**.
- DCG = 1/log2(5) = 0,430677.
- IDCG (9 suku) = 1 + 0,630930 + 0,5 + 0,430677 + 0,386853 + 0,356207 + 0,333333 + 0,315465 + 0,301030 = 4,254495.
- NDCG@10 = 0,430677 / 4,254495 = **0,101229**.

## User C: 1 target, tanpa hit (`AE22B7BMXT7GAGY3XLHICMESNAPA`)
- Target: {B0BYNM15YB}, tidak ada di Top-10.
- Recall@10 = 0/1 = **0**; DCG = 0 → NDCG@10 = **0**.

## Hasil pencocokan
| User | Recall@10 tangan | Recall@10 kode | NDCG@10 tangan | NDCG@10 kode |
|---|---|---|---|---|
| A | 1,000000 | 1,000000 | 0,500000 | 0,500000 |
| B | 0,111111 | 0,111111 | 0,101229 | 0,101229 |
| C | 0 | 0 | 0 | 0 |

Semua cocok (toleransi 1e-6). Test juga memastikan Top-10 dan himpunan target tiap user memang
dihasilkan ulang oleh pipeline dari data validation, serta sama dengan baris per-user pada log E1.
Tidak ada bug metrik yang ditemukan.
