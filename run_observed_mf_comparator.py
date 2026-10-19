"""Continue the observed-only MF comparator at a higher optimizer limit.

Results are written separately and do not alter baseline or sweep files.
"""
import json
import argparse
from pathlib import Path

import numpy as np

import run_baselines as base
import run_mf_rerank_sweep as experiment

MAX_ITER = 1000
OUT = base.ROOT / "outputs" / "w6_followup"
WARM_START = None


def main(output_dir=OUT):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train, valid, candidates, _items = base.load_data()
    users, item_ids, targets, seen, _eligible = base.build_eval(train, valid, candidates)
    popularity = train.groupby("item_id").size().reindex(item_ids).fillna(0).to_numpy(dtype=np.int32)
    tail = popularity <= experiment.TAIL_MAX
    experiment.MAX_OPT_ITER = MAX_ITER
    fit = experiment.observed_only_mf(train, users, item_ids, warm_start=WARM_START)
    scorer = lambda u: (fit["user_factors"][u] @ fit["item_factors"].T
                        + fit["item_bias"] + fit["user_bias"][u] + fit["global_mean"])
    metrics = experiment.score_config("observed_only_mf_1000iter", None, None, scorer,
                                      users, item_ids, targets, seen, popularity, tail)
    fit["fit_details"].update({"max_iterations": MAX_ITER, "score_metrics": metrics})
    np.savez_compressed(output_dir / "observed_only_mf_1000iter_factors.npz",
                        user_factors=fit["user_factors"], item_factors=fit["item_factors"],
                        user_bias=fit["user_bias"], item_bias=fit["item_bias"],
                        global_mean=np.asarray(fit["global_mean"]))
    (output_dir / "observed_only_mf_1000iter.json").write_text(
        json.dumps(fit["fit_details"], indent=2), encoding="utf-8")
    print(json.dumps(fit["fit_details"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit the observed-only MF comparator")
    parser.add_argument("--output-dir", default=str(OUT), help="Use a new path to preserve earlier fit outputs")
    main(parser.parse_args().output_dir)
