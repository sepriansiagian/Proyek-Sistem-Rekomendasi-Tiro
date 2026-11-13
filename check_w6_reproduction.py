"""Re-runs the W6 configurations through the CP3 pipeline and compares with outputs/w6/metrics.json.

This verifies that the CP3 code reproduces W6 rankings and adds Recall@10 for the W6 settings.
It is a pipeline check, not part of the CP3 experiment matrix, and is not appended to the experiment log.
"""
import argparse
import json
from pathlib import Path

import cp3_protocol as P
import run_experiments as R

W6_SPECS = {
    "popularity": ("E1", {"model": "popularity", "fixed": {}, "grid": {}}),
    "mf": ("E2", {"model": "mf", "fixed": {"max_iter": 7000}, "grid": {"factors": [32]}}),
    "tfidf": ("E3", {"model": "tfidf", "fixed": {"min_df": 2}, "grid": {"max_features": [100000], "ngram_max": [1]}}),
    "mf_reranked": ("E4", {"model": "mf_rerank", "fixed": {}, "grid": {"alpha": [1.0], "beta": [6.0]}}),
}


def main(output):
    cfg = json.loads((R.ROOT / "configs" / "cp3_search_space.json").read_text(encoding="utf-8"))
    stored = json.loads((R.ROOT / "outputs" / "w6" / "metrics.json").read_text(encoding="utf-8"))
    data = P.load_development_data()
    report = {}
    for name, (exp, spec) in W6_SPECS.items():
        [(params, recs, _)] = R.run_experiment(exp, spec, cfg, data, factors_for_e4=32)
        metrics, _ = P.evaluate(recs, data[4])
        report[name] = {"params": params, "cp3_metrics": metrics,
                        "w6_ndcg@10": stored[name]["ndcg@10"],
                        "ndcg_abs_diff": abs(metrics["ndcg@10"] - stored[name]["ndcg@10"]),
                        "coverage_abs_diff": abs(metrics["catalog_coverage@10"] - stored[name]["catalog_coverage@10"])}
        print(name, json.dumps({k: report[name][k] for k in ("ndcg_abs_diff", "coverage_abs_diff")}),
              f"recall@10={metrics['recall@10']:.6f}", flush=True)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/cp3/w6_reproduction.json")
    main(parser.parse_args().output)
