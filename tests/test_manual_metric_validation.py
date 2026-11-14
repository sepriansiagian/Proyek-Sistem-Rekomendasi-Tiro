"""Validasi manual Recall@10 / NDCG@10 pada 3 user validation nyata (baseline Popularity, E1).

Nilai harapan dihitung dengan tangan (lihat reports/w10/validasi_manual_metrik.md), bukan dari
keluaran fungsi. Test 1 memakai daftar Top-10 dan target yang ditulis eksplisit; test 2 memastikan
daftar itu memang dihasilkan pipeline dari data validation dan cocok dengan per_user.csv.gz E1.

Run: python -m unittest discover -s tests -v
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cp3_protocol as P  # noqa: E402

TOP10 = ['B09857JRP2', 'B0BCK6L7S5', 'B0BSGM6CQ9', 'B0BPJ4Q6FJ', 'B0BTC9YJ2W',
         'B004XNK7AI', 'B08R5GM6YB', 'B0BHSG5C3G', 'B0BPLFP5P6', 'B08JMQR2JK']

# user_id -> (Top-10 Popularity, target relevan, Recall@10 tangan, NDCG@10 tangan)
CASES = {
    # A: 1 target, hit di rank 3. Recall = 1/1; DCG = 1/log2(4) = 0.5; IDCG = 1/log2(2) = 1.
    "AE3BRGSQ4LSNQUORETJUS7PKBZ7Q": (TOP10, {"B0BSGM6CQ9"}, 1.0, 0.5),
    # B: 9 target, hit di rank 4. Recall = 1/9; DCG = 1/log2(5) = 0.430677;
    #    IDCG = sum_{r=1..9} 1/log2(r+1) = 4.254495; NDCG = 0.430677/4.254495 = 0.101229.
    "AE23LDQTB7L76AP6E6WPBFVYL5DA": (
        TOP10,
        {'B00CGFRJ2Y', 'B07H4LNV2J', 'B07NJBLXBV', 'B08H75RJZV', 'B0B3MWSSYF',
         'B0B6H7J2SN', 'B0BG95DG2H', 'B0BPJ4Q6FJ', 'B0BQF951BC'},
        1 / 9, 0.101229),
    # C: 1 target, tidak ada hit -> DCG = 0 -> keduanya 0.
    "AE22B7BMXT7GAGY3XLHICMESNAPA": (TOP10, {"B0BYNM15YB"}, 0.0, 0.0),
}


class ManualMetricValidation(unittest.TestCase):
    def test_hand_computed_values_match_metric_functions(self):
        for uid, (rec, target, recall, ndcg) in CASES.items():
            with self.subTest(user=uid):
                self.assertAlmostEqual(P.recall_at_k(rec, target, 10), recall, places=6)
                self.assertAlmostEqual(P.ndcg_at_k(rec, target, 10), ndcg, places=6)

    def test_idcg_for_nine_targets(self):
        self.assertAlmostEqual(sum(1 / math.log2(r + 1) for r in range(1, 10)), 4.254495, places=6)

    def test_cases_match_pipeline_and_e1_log(self):
        log = ROOT / "experiments" / "runs" / "E1_popularity_s42" / "per_user.csv.gz"
        if not log.exists():
            self.skipTest("E1 per_user.csv.gz belum ada (jalankan run_experiments.py)")
        import pandas as pd
        import cp3_models as M
        _, _, ids, _, pr = P.load_development_data()
        tie = np.argsort(np.argsort(np.asarray(ids)))
        pop = M.popularity_scores(pr)
        per_user = pd.read_csv(log).set_index("user_id")
        for uid, (rec, target, recall, ndcg) in CASES.items():
            with self.subTest(user=uid):
                u = pr.user_ids.index(uid)
                got = [ids[i] for i in P.top_k(pop, pr.seen[u], 10, tie)]
                self.assertEqual(got, rec)
                self.assertEqual({ids[i] for i in pr.targets[u]}, target)
                self.assertAlmostEqual(per_user.loc[uid, "recall@10"], recall, places=6)
                self.assertAlmostEqual(per_user.loc[uid, "ndcg@10"], ndcg, places=6)


if __name__ == "__main__":
    unittest.main()
