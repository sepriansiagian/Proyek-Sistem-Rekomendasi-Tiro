"""Hand-checkable sanity tests for CP3 metrics, ranking, and leakage guards.

Run: python -m unittest discover -s tests -v
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cp3_protocol as P  # noqa: E402


class MetricTests(unittest.TestCase):
    def test_recall(self):
        self.assertEqual(P.recall_at_k([1, 2, 3], {2, 9}, 10), 0.5)
        self.assertEqual(P.recall_at_k([1, 2, 3], {9}, 10), 0.0)
        self.assertEqual(P.recall_at_k([1, 2, 3], {3}, 2), 0.0)  # outside cutoff

    def test_ndcg_perfect_and_hand_computed(self):
        self.assertAlmostEqual(P.ndcg_at_k([5, 6], {5, 6}, 10), 1.0)
        # one relevant item at rank 2: DCG = 1/log2(3), IDCG = 1
        self.assertAlmostEqual(P.ndcg_at_k([1, 5], {5}, 10), 1 / np.log2(3))
        # two relevant, one hit at rank 1: DCG = 1, IDCG = 1 + 1/log2(3)
        self.assertAlmostEqual(P.ndcg_at_k([5, 1], {5, 6}, 10), 1 / (1 + 1 / np.log2(3)))

    def test_gini(self):
        self.assertAlmostEqual(P.gini(np.array([5, 5, 5, 5])), 0.0)
        self.assertAlmostEqual(P.gini(np.array([0, 0, 0, 12])), 0.75)


class RankingTests(unittest.TestCase):
    def test_top_k_excludes_seen_and_breaks_ties_by_item_order(self):
        scores = np.array([3.0, 9.0, 3.0, 3.0, 1.0])
        tie_rank = np.array([4, 0, 3, 1, 2])  # item 3 sorts before item 2 before item 0
        rec = P.top_k(scores, {1}, 3, tie_rank)
        self.assertEqual(rec.tolist(), [3, 2, 0])
        self.assertNotIn(1, rec.tolist())


class LeakageGuardTests(unittest.TestCase):
    def test_test_split_locked(self):
        with self.assertRaises(PermissionError):
            P.load_split("test")

    def test_crawl_time_aggregates_rejected(self):
        with self.assertRaises(ValueError):
            P.load_items(["item_id", "average_rating"])


if __name__ == "__main__":
    unittest.main()
