# W6 baseline comparison

All primary results below use the same 11,850 evaluation users, 22,753 warm candidates, seen-item filter, and Top-10. Relevance is validation rating ≥4. Popularity is the item's count in training; long-tail means count ≤5. Coverage and exposure are shown as percentages. The primary baseline remains the original MF reranker (`lambda=1`, `beta=6`).

| Method | NDCG@10 | Catalog coverage | Average item popularity | Long-tail exposure | Unique items recommended |
|---|---:|---:|---:|---:|---:|
| Popularity | 0.009221 | 0.079% | 1,361.91 | 0.000% | 18 |
| Truncated SVD MF (32 factors) | 0.007827 | 0.954% | 605.61 | 0.000% | 217 |
| TF-IDF CBF | 0.007293 | 71.700% | 35.72 | 20.566% | 16,314 |
| MF reranking (`lambda=1`, `beta=6`) | 0.007828 | 1.793% | 602.82 | 0.359% | 408 |

## Experimental observed-only MF (not selected)

The 32-factor observed-only MF fit uses L-BFGS-B on observed training ratings only. At 1,000 iterations, its objective decreased from 229,628.49 to 4,841.04, but the optimizer stopped at the iteration limit and reported `success=false`. Validation metrics were NDCG@10 0.002627, coverage 9.968%, average popularity 290.98, and long-tail exposure 15.846% (18,777 slots). It found 0 of 4,599 relevant long-tail target pairs. This is a non-converged experimental comparator, not a replacement baseline.

## Reranking sweep signal

The largest target-relevant long-tail count in the unrestricted sweep was 2 (Long-Tail Recall@10 = 2/4,599 = 0.0435%) at `lambda=2`, `beta=12`: NDCG@10 0.007670, coverage 2.545%, average popularity 582.73, exposure 4.843%. The current `lambda=1`, `beta=6` run reached 425 long-tail slots (0.359% exposure) and 0 relevant hits. Rank-gated bonus sweeps up to original MF rank 2,000 yielded as many as 782 long-tail slots but no relevant hits.

Validation was used for parameter selection, so these results are not independent. No separate test set is available in the supplied data. Full-precision tables are in `outputs/w6/metrics.json` and `outputs/w6_long_tail_relevance/model_comparison.csv`.
