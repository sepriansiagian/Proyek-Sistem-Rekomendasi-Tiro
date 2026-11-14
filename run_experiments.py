"""CP3 experiment runner: executes the experiment matrix on validation and logs every run.

Examples:
    python run_experiments.py --dry-run                 # list every planned run id
    python run_experiments.py --experiments E1 E2       # run and log E1/E2 grids
    python run_experiments.py --experiments E4          # needs E2 in the log (uses selected factors)
    python run_experiments.py --select                  # apply the pre-registered selection rules

Every run writes experiments/runs/<run_id>/{config.json, metrics.json, per_user.csv.gz} and appends
one row to experiments/experiment_log.csv with the git commit, data fingerprint, and seed.
Test data is never loaded (see cp3_protocol.LOCKED_SPLITS).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import cp3_models as models
import cp3_protocol as P

ROOT = Path(__file__).resolve().parent
EXP_DIR = ROOT / "experiments"
LOG = EXP_DIR / "experiment_log.csv"
SELECTION = EXP_DIR / "selection.json"
PARAM_ABBREV = {"factors": "f", "max_features": "mf", "ngram_max": "ng", "alpha": "a", "beta": "b",
                "mf_factors": "f"}
BATCH = 128


# ------------------------------------------------------------ provenance ---

def git_info():
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""
    status = run("status", "--porcelain", "--untracked-files=no")
    return {"git_commit": run("rev-parse", "--short", "HEAD") or "unknown", "git_dirty": bool(status)}


def data_fingerprint():
    digest = hashlib.sha256()
    for name in (P.PROTOCOL["train_file"], P.PROTOCOL["validation_file"], P.PROTOCOL["candidate_file"]):
        digest.update((P.DATA / name).read_bytes())
    return digest.hexdigest()[:12]


def run_id(exp, model, params, seed):
    parts = [f"{PARAM_ABBREV.get(k, k)}{v}" for k, v in params.items()]
    return "_".join([exp, model, *parts, f"s{seed}"])


def expand(spec):
    keys = list(spec["grid"])
    return [dict(zip(keys, values)) for values in itertools.product(*(spec["grid"][k] for k in keys))] or [{}]


# --------------------------------------------------------------- ranking ---

def rank_users(protocol, score_fn, keys):
    """score_fn(user_idx_array) -> {key: dense score matrix}; returns {key: list of top-k arrays}."""
    k = P.PROTOCOL["top_k"]
    tie_rank = np.argsort(np.argsort(np.asarray(protocol.item_ids)))
    recs = {key: [] for key in keys}
    for start in range(0, len(protocol.user_ids), BATCH):
        idx = np.arange(start, min(start + BATCH, len(protocol.user_ids)))
        batch_scores = score_fn(idx)
        for key in keys:
            mat = batch_scores[key]
            for r, u in enumerate(idx):
                recs[key].append(P.top_k(mat[r] if mat.ndim == 2 else mat, protocol.seen[u], k, tie_rank))
    return recs


def run_experiment(exp, spec, cfg, data, factors_for_e4=None):
    """Returns list of (params, recs, extra_info) for every grid point of one experiment."""
    train, _, _, items, protocol = data
    seed, grid = cfg["seed"], expand(spec)
    model = spec["model"]
    if model == "popularity":
        pop = models.popularity_scores(protocol)
        return [({}, rank_users(protocol, lambda idx: {None: pop}, [None])[None], {})]
    if model == "mf":
        out = []
        for params in grid:
            U, V = models.train_mf(train, protocol, params["factors"], seed, spec["fixed"]["max_iter"])
            out.append((params, rank_users(protocol, lambda idx: {None: U[idx] @ V.T}, [None])[None], {}))
        return out
    if model == "tfidf":
        out = []
        for params in grid:
            X, vocab = models.tfidf_item_matrix(items, protocol.item_ids, params["max_features"],
                                                params["ngram_max"], spec["fixed"]["min_df"])
            prof = models.user_content_profiles(train, protocol, X)
            fn = lambda idx: {None: (prof[idx] @ X.T).toarray()}
            out.append((params, rank_users(protocol, fn, [None])[None], {"vocabulary_size": vocab}))
        return out
    if model == "mf_rerank":
        if factors_for_e4 is None:
            raise RuntimeError("E4 needs the selected E2 factors: run E2 first or pass --mf-factors")
        U, V = models.train_mf(train, protocol, factors_for_e4, seed, cfg["experiments"]["E2"]["fixed"]["max_iter"])
        keys = [tuple(p.items()) for p in grid]

        def fn(idx):
            mf = U[idx] @ V.T
            seen = [protocol.seen[u] for u in idx]
            # z-scores do not depend on alpha/beta, so compute them once per batch.
            z = models.rerank_scores(mf, seen, protocol, 0.0, 0.0)
            pop_norm = np.log1p(protocol.popularity) / np.log1p(max(int(protocol.popularity.max()), 1))
            return {key: z - dict(key)["alpha"] * pop_norm + dict(key)["beta"] * protocol.tail for key in keys}
        recs = rank_users(protocol, fn, keys)
        return [(dict(mf_factors=factors_for_e4, **dict(key)), recs[key], {}) for key in keys]
    raise ValueError(f"Unknown model {model}")


# --------------------------------------------------------------- logging ---

def log_run(exp, spec, cfg, params, recs, extra, protocol, provenance, started, duration):
    metrics, per_user = P.evaluate(recs, protocol)
    rid = run_id(exp, spec["model"], params, cfg["seed"])
    run_dir = EXP_DIR / "runs" / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    run_cfg = {"run_id": rid, "experiment": exp, "model": spec["model"], "params": params,
               "fixed": spec["fixed"], "seed": cfg["seed"], "split": "validation",
               "config_version": cfg["version"], "protocol": P.PROTOCOL, **provenance, **extra,
               "started_at": started}
    (run_dir / "config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    per_user.to_csv(run_dir / "per_user.csv.gz", index=False)
    row = {"run_id": rid, "experiment": exp, "model": spec["model"], "params": json.dumps(params),
           "seed": cfg["seed"], "split": "validation", "config_version": cfg["version"], **provenance,
           "started_at": started, "duration_s": round(duration, 1), **metrics}
    EXP_DIR.mkdir(exist_ok=True)
    pd.DataFrame([row]).to_csv(LOG, mode="a", header=not LOG.exists(), index=False)
    print(f"[{rid}] recall@10={metrics['recall@10']:.5f} ndcg@10={metrics['ndcg@10']:.5f} "
          f"coverage={metrics['catalog_coverage@10']:.4f} tail={metrics['long_tail_exposure@10']:.4f}", flush=True)


# ------------------------------------------------------------- selection ---

def latest_runs(exp):
    if not LOG.exists():
        return pd.DataFrame()
    log = pd.read_csv(LOG)
    log = log[log.experiment == exp]
    return log.drop_duplicates("run_id", keep="last")


def select(exp, cfg):
    runs = latest_runs(exp)
    if runs.empty:
        return None
    runs = runs.assign(params_dict=runs.params.map(json.loads))
    if exp in ("E1",):
        best = runs.iloc[0]
    elif exp == "E2":
        runs = runs.assign(f=runs.params_dict.map(lambda p: p["factors"]))
        best = runs.sort_values(["ndcg@10", "recall@10", "f"], ascending=[False, False, True]).iloc[0]
    elif exp == "E3":
        runs = runs.assign(mf=runs.params_dict.map(lambda p: p["max_features"]),
                           ng=runs.params_dict.map(lambda p: p["ngram_max"]))
        best = runs.sort_values(["ndcg@10", "recall@10", "mf", "ng"],
                                ascending=[False, False, True, True]).iloc[0]
    elif exp == "E4":
        e2 = select("E2", cfg)
        floor = 0.95 * e2["metrics"]["ndcg@10"]
        runs = runs.assign(strength=runs.params_dict.map(lambda p: p["alpha"] + p["beta"]))
        ok = runs[runs["ndcg@10"] >= floor]
        best = ok.sort_values(["long_tail_exposure@10", "catalog_coverage@10", "strength"],
                              ascending=[False, False, True]).iloc[0]
    else:
        raise ValueError(exp)
    metric_cols = [c for c in runs.columns if "@" in c or c == "users_evaluated"]
    return {"run_id": best.run_id, "params": best.params_dict, "rule": cfg["selection_rules"][exp],
            "git_commit": best.git_commit, "metrics": {c: best[c] for c in metric_cols}}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/cp3_search_space.json")
    parser.add_argument("--experiments", nargs="+", default=[])
    parser.add_argument("--mf-factors", type=int, help="override the E2-selected factors used by E4")
    parser.add_argument("--dry-run", action="store_true", help="print the planned run ids only")
    parser.add_argument("--select", action="store_true", help="write experiments/selection.json")
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    np.random.seed(cfg["seed"])

    if args.dry_run:
        chosen = select("E2", cfg)
        e2_factors = args.mf_factors or (chosen["params"]["factors"] if chosen else "SEL")
        for exp, spec in cfg["experiments"].items():
            for params in expand(spec):
                if spec["model"] == "mf_rerank":
                    params = dict(mf_factors=e2_factors, **params)
                print(run_id(exp, spec["model"], params, cfg["seed"]))
        return
    if args.experiments:
        provenance = dict(git_info(), data_sha=data_fingerprint())
        if provenance["git_dirty"]:
            print("WARNING: tracked files have uncommitted changes; commit before runs you report.", flush=True)
        data = P.load_development_data()
        for exp in args.experiments:
            spec = cfg["experiments"][exp]
            factors = args.mf_factors
            if spec["model"] == "mf_rerank" and factors is None:
                chosen = select("E2", cfg)
                factors = chosen["params"]["factors"] if chosen else None
            started, t0 = datetime.now(timezone.utc).isoformat(timespec="seconds"), time.time()
            results = run_experiment(exp, spec, cfg, data, factors)
            duration = (time.time() - t0) / len(results)
            for params, recs, extra in results:
                log_run(exp, spec, cfg, params, recs, extra, data[4], provenance, started, duration)
    if args.select:
        selection = {exp: select(exp, cfg) for exp in cfg["experiments"]}
        SELECTION.write_text(json.dumps(selection, indent=2, default=str), encoding="utf-8")
        print(json.dumps(selection, indent=2, default=str))


if __name__ == "__main__":
    main()
