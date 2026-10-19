"""Recompute baseline rankings and verify ranking invariants without overwriting prior artifacts."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_baselines as base


def main(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train, valid, candidates, items = base.load_data()
    users, item_ids, targets, seen, eligible = base.build_eval(train, valid, candidates)
    popularity = train.groupby("item_id").size().reindex(item_ids).fillna(0).to_numpy(dtype=np.int32)
    mf_u, mf_v = base.train_mf(train, users, item_ids)
    item_tfidf, _ = base.tfidf_item_matrix(items, item_ids)
    profiles = base.build_user_content_profiles(train, users, item_ids, item_tfidf)
    recommendations, user_metrics, tail = base.recommend_all(
        users, item_ids, targets, seen, popularity, mf_u, mf_v, profiles, item_tfidf)
    recalculated = base.summarize(recommendations, user_metrics, item_ids, popularity, tail)
    stored = json.loads((base.OUT / "metrics.json").read_text(encoding="utf-8"))
    checks, example_rows = {}, []
    for model, lists in recommendations.items():
        lengths = [len(row) for row in lists]
        duplicate_users = sum(len(set(row.tolist())) != len(row) for row in lists)
        seen_recs = sum(bool(set(row.tolist()) & seen[u]) for u, row in enumerate(lists))
        rec_flat = np.concatenate(lists)
        metric_diffs = {name: abs(recalculated[model][name] - stored[model][name])
                        for name in ("ndcg@10", "catalog_coverage@10",
                                     "average_item_popularity@10", "long_tail_exposure@10")}
        checks[model] = {"users": len(lists), "recommended_slots": int(len(rec_flat)),
                         "min_recommendations": int(min(lengths)), "max_recommendations": int(max(lengths)),
                         "users_with_10": int(sum(n == 10 for n in lengths)),
                         "users_with_duplicate_recommendations": int(duplicate_users),
                         "users_with_seen_item_recommended": int(seen_recs),
                         "unique_catalog_items": int(len(np.unique(rec_flat))),
                         "stored_metric_max_abs_diff": metric_diffs}
        for u in range(min(10, len(users))):
            for rank, idx in enumerate(lists[u], 1):
                example_rows.append({"model": model, "user_id": users[u], "rank": rank,
                                     "item_id": item_ids[idx], "train_popularity": int(popularity[idx]),
                                     "is_long_tail": bool(tail[idx]), "validation_relevant": idx in targets[u]})
    shared = {"candidate_count": len(item_ids), "candidate_ids_unique": len(set(item_ids)) == len(item_ids),
              "candidate_set_equals_train_item_set": set(item_ids) == set(train.item_id),
              "mf_fit_interactions": int(len(train)), "mf_fit_users": int(train.user_id.nunique()),
              "validation_train_pair_overlap": int(len(set(zip(train.user_id, train.item_id)) & set(zip(valid.user_id, valid.item_id)))),
              "evaluation_users": len(users), "eligible_validation_relevant_rows": int(len(eligible)),
              "long_tail_threshold": base.CONFIG["long_tail_max_train_interactions"],
              "tail_candidate_count": int(tail.sum()),
              "relevant_tail_targets_in_eval": int(sum(sum(bool(tail[i]) for i in t) for t in targets))}
    report = {"checks": checks, "shared": shared, "metrics_recalculated": recalculated,
              "baseline_metrics_match_within_1e-12": all(
                  max(v["stored_metric_max_abs_diff"].values()) <= 1e-12 for v in checks.values())}
    (output_dir / "sanity_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(example_rows).to_csv(output_dir / "sanity_audit_examples.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/w6_sanity_rerun",
                        help="Use a new path so existing audit outputs remain untouched")
    main(parser.parse_args().output_dir)
