"""Validation-only, non-destructive long-tail relevance reranking sweep."""
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import run_baselines as base
import run_mf_rerank_sweep as prior

OUT = base.ROOT / "outputs" / "w6_long_tail_relevance"
K = 10
TAIL_MAX = 5
REL_MIN = 4.0
GATES = [50, 100, 250, 500, 1000, 2000]
BONUSES = [1.0, 2.0, 4.0, 8.0]
PENALTY = 1.0


def main(output_dir=OUT):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train, valid, candidates, _ = base.load_data()
    old = base.CONFIG["relevant_rating_min"]
    base.CONFIG["relevant_rating_min"] = REL_MIN
    users, item_ids, targets, seen, eligible = base.build_eval(train, valid, candidates)
    base.CONFIG["relevant_rating_min"] = old
    pop = train.groupby("item_id").size().reindex(item_ids).fillna(0).to_numpy(np.int32)
    tail = pop <= TAIL_MAX
    tie_ids = np.asarray(item_ids, dtype=str)
    tie_order = np.argsort(tie_ids)
    pop_norm = np.log1p(pop) / np.log1p(max(int(pop.max()), 1))
    svd_u, svd_v = base.train_mf(train, users, item_ids)
    # Same candidate universe and seen-item filter as all preceding evaluations.
    score_mat = svd_u @ svd_v.T
    ranks = np.empty_like(score_mat, dtype=np.int32)
    for u in range(len(users)):
        adjusted = score_mat[u].copy()
        if seen[u]:
            adjusted[list(seen[u])] = -np.inf
        order = np.lexsort((tie_ids, -adjusted))
        ranks[u, order] = np.arange(len(item_ids))

    target_tail_total = int(sum(sum(tail[i] for i in t) for t in targets))
    def metric(name, rec_lists):
        ndcgs = []
        flat = []
        slots = hits = 0
        for u, rec in enumerate(rec_lists):
            ndcgs.append(prior.ndcg(rec, targets[u], K))
            flat.extend(rec)
            slots += int(tail[rec].sum())
            hits += sum(bool(tail[i]) and i in targets[u] for i in rec)
        flat = np.asarray(flat, dtype=np.int64)
        return {"config": name, "ndcg@10": float(np.mean(ndcgs)),
                "catalog_coverage@10": float(len(np.unique(flat)) / len(item_ids)),
                "average_item_popularity@10": float(pop[flat].mean()),
                "long_tail_exposure@10": float(slots / len(flat)), "long_tail_slots": slots,
                "relevant_long_tail_hits@10": hits,
                "long_tail_recall@10": float(hits / target_tail_total) if target_tail_total else 0.0,
                "relevant_long_tail_targets": target_tail_total,
                "users_evaluated": len(users), "unique_recommended_items": int(len(np.unique(flat)))}

    raw_recs = [base.top_indices(score_mat[u].copy(), seen[u], K, tie_ids, tie_order)
                for u in range(len(users))]
    results = [metric("svd_mf_reference", raw_recs)]
    configs = []
    for gate in GATES:
        for bonus in BONUSES:
            recs = []
            for u in range(len(users)):
                raw = score_mat[u]
                available = np.ones(len(item_ids), dtype=bool)
                if seen[u]:
                    available[list(seen[u])] = False
                z = (raw - raw[available].mean()) / (raw[available].std() or 1.0)
                eligible_tail = tail & (ranks[u] < gate)
                final = z - PENALTY * pop_norm + bonus * eligible_tail
                recs.append(base.top_indices(final, seen[u], K, tie_ids, tie_order))
            name = f"gate{gate}_bonus{bonus:g}_penalty{PENALTY:g}"
            row = metric(name, recs)
            row.update({"mf_rank_gate": gate, "tail_bonus": bonus, "popularity_penalty": PENALTY})
            results.append(row)
            configs.append({"mf_rank_gate": gate, "tail_bonus": bonus, "popularity_penalty": PENALTY})

    frame = pd.DataFrame(results)
    ref = frame.iloc[0]["ndcg@10"]
    frame["passes_ndcg_5pct_limit"] = frame["ndcg@10"] >= ref * 0.95
    frame["improves_long_tail_recall"] = frame["long_tail_recall@10"] > frame.iloc[0]["long_tail_recall@10"]
    frame.to_csv(output_dir / "long_tail_relevance_sweep.csv", index=False)
    protocol = {"candidate_count": len(item_ids), "users": len(users), "candidate_policy": "candidate_items_warm minus user's training interactions",
                "relevance": f"validation rating >= {REL_MIN}", "top_k": K,
                "long_tail_definition": f"training interaction count <= {TAIL_MAX}",
                "long_tail_recall_definition": "relevant validation long-tail user-item targets retrieved in Top-10 / all relevant validation long-tail user-item targets",
                "target_long_tail_pairs": target_tail_total, "popularity_source": "training interaction count",
                "formula": "zscore(SVD MF) - 1.0 * normalized log popularity + bonus * long-tail indicator gated by original MF rank < gate",
                "gates": GATES, "bonuses": BONUSES, "success_criterion": "NDCG@10 >= 95% SVD reference and long-tail recall@10 above SVD reference",
                "validation_used_for_selection": True, "independent_test_available": False,
                "baseline_outputs_untouched": True, "relevant_validation_rows": len(eligible)}
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(frame.to_string(index=False))
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run validation-only long-tail relevance sweep")
    parser.add_argument("--output-dir", default=str(OUT), help="Use a new path to preserve earlier sweeps")
    main(parser.parse_args().output_dir)
