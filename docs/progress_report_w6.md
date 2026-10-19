# Progress Report W6 — draft

## Objective and completed work

The selected research track studies popularity bias and long-tail exposure: whether reranking can increase exposure to less popular products while keeping relevance. Dataset audit and a consistent evaluation protocol were completed. Four primary methods were implemented and compared: popularity ranking, rank-32 Truncated SVD matrix factorization, TF-IDF content-based filtering, and popularity-aware reranking of the MF ranking. Sanity checks confirm 11,850 users each receive 10 unique items, with no training-seen items recommended. All methods use the same 22,753 warm candidates and training-based item popularity.

## Main validation results

| Method | NDCG@10 | Coverage | Average item popularity | Long-tail exposure |
|---|---:|---:|---:|---:|
| Popularity | 0.009221 | 0.079% | 1,361.91 | 0.000% |
| Truncated SVD MF | 0.007827 | 0.954% | 605.61 | 0.000% |
| TF-IDF CBF | 0.007293 | 71.700% | 35.72 | 20.566% |
| MF reranking (`lambda=1`, `beta=6`) | 0.007828 | 1.793% | 602.82 | 0.359% |

The main reranker preserved NDCG relative to SVD while increasing long-tail exposure from 0% to 0.359% (425 recommendation slots). However, none of those 425 long-tail slots matched a relevant validation target. In the controlled unrestricted sweep, the strongest reported relevant-target result was 2 hits among 4,599 relevant long-tail targets (Long-Tail Recall@10 = 0.0435%) at `lambda=2`, `beta=12`; NDCG@10 was 0.007670 and exposure 4.843%. This satisfies the previously specified 5% relative NDCG drop bound but yields very low long-tail relevance. Rank-gated bonus experiments also increased exposure without retrieving a relevant long-tail target.

An observed-only MF comparator was fit on observed training ratings only. At 1,000 iterations its objective fell from 229,628.49 to 4,841.04, but optimization stopped at the iteration limit, so convergence was not achieved. Its validation NDCG@10 was 0.002627; it remains an experimental comparator and does not replace the Truncated SVD baseline.

## Interpretation and limitations

The results show that exposure can be raised by reranking, but the added long-tail recommendations rarely correspond to validation-relevant items. Relevance-aware rank gates up to rank 2,000 did not resolve this. Validation was used to select reranking parameters, so sweep results must not be described as independent evaluation. The supplied files contain no independent test split. The observed-only optimizer needs further convergence work, and an independent test evaluation is needed before final performance claims.

## Reproducibility and artifacts

Code is in `run_baselines.py`, `run_mf_rerank_sweep.py`, `run_long_tail_relevance_sweep.py`, and `run_observed_mf_comparator.py`. Baseline metrics, configuration, per-user NDCG, Top-10 examples and sanity checks are under `outputs/w6/`; MF sweeps and observed-only fit records are under `outputs/w6_followup/`; relevance-gated sweep and merged comparisons are under `outputs/w6_long_tail_relevance/`. README commands support a fresh `--output-dir` so old output folders remain intact.
