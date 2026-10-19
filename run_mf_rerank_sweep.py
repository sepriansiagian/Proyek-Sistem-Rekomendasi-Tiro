"""Non-destructive MF method audit and validation reranking sweep.

This writes only to outputs/w6_followup and never replaces outputs/w6.
"""
from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import run_baselines as base

OUT = base.ROOT / "outputs" / "w6_followup"
K = 10
REL_MIN = 4.0
TAIL_MAX = 5
SEED = 42
FACTOR_COUNT = 32
LAMBDA_BIAS = 0.1
LAMBDA_FACTORS = 0.1
MAX_OPT_ITER = 25
NDGC_RELATIVE_DROP_LIMIT = 0.05
SWEEP = [
    {"name": "svd_mf_reference", "penalty": None, "tail_bonus": None},
    {"name": "penalty_1_bonus_0", "penalty": 1.0, "tail_bonus": 0.0},
    {"name": "penalty_2_bonus_0", "penalty": 2.0, "tail_bonus": 0.0},
    {"name": "penalty_0_bonus_4", "penalty": 0.0, "tail_bonus": 4.0},
    {"name": "penalty_0_bonus_8", "penalty": 0.0, "tail_bonus": 8.0},
    {"name": "penalty_0_bonus_12", "penalty": 0.0, "tail_bonus": 12.0},
    {"name": "penalty_1_bonus_6_current", "penalty": 1.0, "tail_bonus": 6.0},
    {"name": "penalty_1_bonus_10", "penalty": 1.0, "tail_bonus": 10.0},
    {"name": "penalty_2_bonus_12", "penalty": 2.0, "tail_bonus": 12.0},
]


def observed_only_mf(train, users, item_ids, warm_start=None):
    """Fit biased explicit MF by minimizing error only over observed train rows."""
    train_users = sorted(train.user_id.unique())
    user_index = {u: i for i, u in enumerate(train_users)}
    item_index = {item: i for i, item in enumerate(item_ids)}
    rows = train.user_id.map(user_index).to_numpy(dtype=np.int32)
    cols = train.item_id.map(item_index).to_numpy(dtype=np.int32)
    ratings = train.rating.to_numpy(dtype=np.float64)
    n_users, n_items, f = len(train_users), len(item_ids), FACTOR_COUNT
    global_mean = float(ratings.mean())
    rng = np.random.default_rng(SEED)
    p0 = rng.normal(0.0, 0.03, size=(n_users, f))
    q0 = rng.normal(0.0, 0.03, size=(n_items, f))
    bu0 = np.zeros(n_users, dtype=np.float64)
    bi0 = np.zeros(n_items, dtype=np.float64)
    if warm_start is not None:
        previous = np.load(warm_start)
        for row, user in enumerate(users):
            if user in user_index:
                p0[user_index[user]] = previous["user_factors"][row]
                bu0[user_index[user]] = previous["user_bias"][row]
        q0[:] = previous["item_factors"]
        bi0[:] = previous["item_bias"]
    offsets = (n_users * f, n_items * f, n_users, n_items)
    x0 = np.concatenate((p0.ravel(), q0.ravel(), bu0, bi0))

    def unpack(x):
        a, b, c, _ = offsets
        p = x[:a].reshape(n_users, f)
        q = x[a:a+b].reshape(n_items, f)
        bu = x[a+b:a+b+c]
        bi = x[a+b+c:]
        return p, q, bu, bi

    def objective(x):
        p, q, bu, bi = unpack(x)
        pu, qi = p[rows], q[cols]
        residual = global_mean + bu[rows] + bi[cols] + np.einsum("ij,ij->i", pu, qi) - ratings
        loss = 0.5 * residual.dot(residual)
        loss += 0.5 * LAMBDA_FACTORS * (np.square(p).sum() + np.square(q).sum())
        loss += 0.5 * LAMBDA_BIAS * (np.square(bu).sum() + np.square(bi).sum())
        gp = LAMBDA_FACTORS * p
        gq = LAMBDA_FACTORS * q
        np.add.at(gp, rows, residual[:, None] * qi)
        np.add.at(gq, cols, residual[:, None] * pu)
        gbu = LAMBDA_BIAS * bu + np.bincount(rows, weights=residual, minlength=n_users)
        gbi = LAMBDA_BIAS * bi + np.bincount(cols, weights=residual, minlength=n_items)
        grad = np.concatenate((gp.ravel(), gq.ravel(), gbu, gbi))
        return loss, grad

    start_loss = objective(x0)[0]
    fit = minimize(objective, x0, method="L-BFGS-B", jac=True,
                   options={"maxiter": MAX_OPT_ITER, "ftol": 1e-7, "maxcor": 5})
    p, q, bu, bi = unpack(fit.x)
    eval_rows = np.fromiter((user_index[u] for u in users), dtype=np.int64)
    return {
        "user_factors": p[eval_rows].astype(np.float32),
        "item_factors": q.astype(np.float32),
        "user_bias": bu[eval_rows].astype(np.float32),
        "item_bias": bi.astype(np.float32),
        "global_mean": global_mean,
        "fit_details": {"success": bool(fit.success), "message": str(fit.message),
                        "iterations": int(fit.nit), "objective_initial": float(start_loss),
                        "objective_final": float(fit.fun), "observed_train_rows": int(len(ratings)),
                        "unobserved_cells_in_objective": 0,
                        "objective": "0.5*sum_observed((r_ui - mu - b_u - b_i - p_u.q_i)^2) + L2 regularization",
                        "warm_start": str(warm_start) if warm_start is not None else None},
    }


def ndcg(recs, target, k=K):
    dcg = sum(1.0 / np.log2(rank + 2) for rank, x in enumerate(recs) if x in target)
    ideal = sum(1.0 / np.log2(rank + 2) for rank in range(min(k, len(target))))
    return dcg / ideal if ideal else 0.0


def score_config(name, penalty, bonus, score_for_user, users, item_ids, targets, seen, popularity, tail):
    n_items = len(item_ids)
    pop_norm = np.log1p(popularity) / np.log1p(max(int(popularity.max()), 1))
    tie_ids = np.asarray(item_ids, dtype=str)
    tie_order = np.argsort(tie_ids)
    per_user, all_items = [], []
    slots = relevant_tail_hits = 0
    for u in range(len(users)):
        raw = score_for_user(u)
        available = np.ones(n_items, dtype=bool)
        if seen[u]:
            available[list(seen[u])] = False
        sigma = float(raw[available].std()) or 1.0
        z = (raw - float(raw[available].mean())) / sigma
        if penalty is None:
            scores = raw.copy()
        else:
            scores = z - penalty * pop_norm + bonus * tail
        rec = base.top_indices(scores, seen[u], K, tie_ids, tie_order)
        per_user.append(ndcg(rec, targets[u]))
        all_items.extend(rec.tolist())
        slots += int(tail[rec].sum())
        relevant_tail_hits += sum(bool(tail[x]) and x in targets[u] for x in rec)
    flat = np.asarray(all_items, dtype=np.int64)
    unique = np.unique(flat)
    return {
        "config": name, "penalty": penalty, "tail_bonus": bonus,
        "ndcg@10": float(np.mean(per_user)),
        "catalog_coverage@10": float(len(unique) / len(item_ids)),
        "average_item_popularity@10": float(popularity[flat].mean()),
        "long_tail_exposure@10": float(slots / len(flat)),
        "long_tail_slots": int(slots),
        "relevant_long_tail_slots": int(relevant_tail_hits),
        "users_evaluated": len(users),
        "unique_recommended_items": int(len(unique)),
    }


def main(output_dir=OUT):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train, valid, candidates, _items = base.load_data()
    # Use the baseline protocol and the exact same validation user/candidate/seen definitions.
    old_rel = base.CONFIG["relevant_rating_min"]
    base.CONFIG["relevant_rating_min"] = REL_MIN
    users, item_ids, targets, seen, eligible = base.build_eval(train, valid, candidates)
    base.CONFIG["relevant_rating_min"] = old_rel
    popularity = train.groupby("item_id").size().reindex(item_ids).fillna(0).to_numpy(dtype=np.int32)
    tail = popularity <= TAIL_MAX
    protocol = {
        "output_directory": str(output_dir),
        "baseline_output_directory_untouched": str(base.OUT),
        "candidate_count": len(item_ids), "evaluation_users": len(users),
        "relevant_validation_rows": len(eligible), "relevance_rating_min": REL_MIN,
        "top_k": K, "long_tail_definition": f"train item interaction count <= {TAIL_MAX}",
        "success_criterion": f"NDCG@10 >= SVD MF reference * {1-NDGC_RELATIVE_DROP_LIMIT:.2f} and long-tail exposure > SVD MF reference",
        "max_relative_ndcg_drop": NDGC_RELATIVE_DROP_LIMIT,
        "test_set_available": False,
        "reranking_formula": "zscore(MF) - lambda * normalized_log_train_popularity + beta * long_tail_indicator",
        "sweep": SWEEP,
        "observed_only_mf": {"factors": FACTOR_COUNT, "optimizer": "L-BFGS-B", "max_iterations": MAX_OPT_ITER,
                             "lambda_factors": LAMBDA_FACTORS, "lambda_bias": LAMBDA_BIAS,
                             "loss_scope": "observed training rows only; missing pairs contribute no rating error"},
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(f"Protocol saved; {len(users):,} users and {len(item_ids):,} candidates", flush=True)

    # Reproduce the original, zero-filled Truncated SVD MF reference without writing to baseline outputs.
    svd_u, svd_v = base.train_mf(train, users, item_ids)
    svd_score = lambda u: svd_u[u] @ svd_v.T
    svd_reference = score_config("svd_mf_reference", None, None, svd_score, users, item_ids,
                                 targets, seen, popularity, tail)
    print("SVD reference ranked", flush=True)

    # Observed-only comparator: fit an explicit rating loss only on non-missing train pairs.
    observed = observed_only_mf(train, users, item_ids)
    observed_score = lambda u: (observed["user_factors"][u] @ observed["item_factors"].T
                                + observed["item_bias"] + observed["user_bias"][u] + observed["global_mean"])
    observed_reference = score_config("observed_only_mf_reference", None, None, observed_score, users,
                                      item_ids, targets, seen, popularity, tail)
    print("Observed-only MF ranked", flush=True)

    # Stream one SVD user score vector at a time across each controlled parameter setting.
    sweep_results = []
    # score_config is deliberately used per configuration so the implementation is identical.
    for cfg in SWEEP:
        if cfg["name"] == "svd_mf_reference":
            result = dict(svd_reference)
        else:
            result = score_config(cfg["name"], cfg["penalty"], cfg["tail_bonus"], svd_score,
                                  users, item_ids, targets, seen, popularity, tail)
        result["within_5pct_ndcg_limit"] = result["ndcg@10"] >= svd_reference["ndcg@10"] * (1-NDGC_RELATIVE_DROP_LIMIT)
        result["success_criterion_met"] = result["within_5pct_ndcg_limit"] and result["long_tail_exposure@10"] > svd_reference["long_tail_exposure@10"]
        sweep_results.append(result)
        print(f"Finished {cfg['name']}: NDCG={result['ndcg@10']:.6f}, tail={result['long_tail_exposure@10']:.4%}", flush=True)

    compare = [dict(svd_reference, comparison="SVD zero-filled MF")]
    compare.append(dict(observed_reference, comparison="Observed-only MF"))
    pd.DataFrame(sweep_results).to_csv(output_dir / "reranking_sweep.csv", index=False)
    pd.DataFrame(compare).to_csv(output_dir / "mf_method_comparison.csv", index=False)
    observed["fit_details"]["score_metrics"] = observed_reference
    (output_dir / "observed_only_mf_fit.json").write_text(json.dumps(observed["fit_details"], indent=2), encoding="utf-8")
    (output_dir / "results.json").write_text(json.dumps({"svd_reference": svd_reference,
        "observed_only_mf": observed_reference, "sweep": sweep_results}, indent=2), encoding="utf-8")
    print("Saved follow-up results without modifying outputs/w6", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MF and reranking validation sweep")
    parser.add_argument("--output-dir", default=str(OUT), help="Use a new path to preserve earlier sweeps")
    main(parser.parse_args().output_dir)
