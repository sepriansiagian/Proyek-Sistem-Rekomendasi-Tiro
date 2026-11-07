"""CP3 shared warm-start evaluation protocol, leakage guards, and metrics.

Every CP3 experiment imports this module so that all models share the same
split usage, evaluable users, candidate universe, and metric definitions.

Run directly to write a protocol/leakage report:
    python cp3_protocol.py --output outputs/cp3/protocol_report.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

PROTOCOL = {
    "split": "official absolute-timestamp split (no reshuffle, no new split)",
    "train_file": "interactions_train.csv",
    "validation_file": "interactions_validation.csv",
    "candidate_file": "candidate_items_warm.csv",
    "item_file": "items.csv",
    "relevant_rating_min": 4.0,
    "top_k": 10,
    "long_tail_max_train_interactions": 5,
    "head_item_share": 0.20,
    "candidate_policy": "shared warm candidate universe minus each user's training items",
    "evaluable_user_policy": "user has >=1 training interaction and >=1 relevant validation item in the warm candidate universe",
}
SPLIT_FILES = {"train": PROTOCOL["train_file"], "validation": PROTOCOL["validation_file"]}
# Test stays locked during development; it is opened only for the final W13 run.
LOCKED_SPLITS = {"test"}
ALLOWED_ITEM_FIELDS = ["item_id", "title", "content_text_basic"]
FORBIDDEN_ITEM_FIELDS = {"average_rating", "rating_number"}


def load_split(name: str) -> pd.DataFrame:
    if name in LOCKED_SPLITS:
        raise PermissionError(f"Split '{name}' is locked until the final W13 evaluation.")
    if name not in SPLIT_FILES:
        raise ValueError(f"Unknown split '{name}'")
    frame = pd.read_csv(DATA / SPLIT_FILES[name], dtype={"user_id": str, "item_id": str})
    return frame


def load_candidates() -> list[str]:
    return pd.read_csv(DATA / PROTOCOL["candidate_file"], dtype={"item_id": str})["item_id"].tolist()


def load_items(fields=ALLOWED_ITEM_FIELDS) -> pd.DataFrame:
    blocked = FORBIDDEN_ITEM_FIELDS & set(fields)
    if blocked:
        raise ValueError(f"Crawl-time aggregate fields are not allowed as features: {sorted(blocked)}")
    return pd.read_csv(DATA / PROTOCOL["item_file"], usecols=list(fields), dtype={"item_id": str})


@dataclass
class Protocol:
    user_ids: list[str]          # evaluable users, sorted
    item_ids: list[str]          # shared candidate universe, file order
    targets: list[set[int]]      # relevant validation items per evaluable user (item indices)
    seen: list[set[int]]         # training items per evaluable user (excluded from ranking)
    popularity: np.ndarray       # training interaction count per candidate item
    tail: np.ndarray             # bool, long-tail item
    head: np.ndarray             # bool, top-20% most popular candidate items
    relevant_rows: int


def build_protocol(train: pd.DataFrame, valid: pd.DataFrame, item_ids: list[str]) -> Protocol:
    item_index = {item: i for i, item in enumerate(item_ids)}
    relevant = valid[(valid.rating >= PROTOCOL["relevant_rating_min"]) & valid.item_id.isin(item_index)]
    user_ids = sorted(set(train.user_id) & set(relevant.user_id))
    user_index = {u: i for i, u in enumerate(user_ids)}
    eligible = relevant[relevant.user_id.isin(user_index)]
    targets = [set() for _ in user_ids]
    for u, i in zip(eligible.user_id, eligible.item_id):
        targets[user_index[u]].add(item_index[i])
    seen = [set() for _ in user_ids]
    known = train[train.user_id.isin(user_index) & train.item_id.isin(item_index)]
    for u, i in zip(known.user_id, known.item_id):
        seen[user_index[u]].add(item_index[i])
    popularity = train.groupby("item_id").size().reindex(item_ids).fillna(0).to_numpy(dtype=np.int64)
    tail = popularity <= PROTOCOL["long_tail_max_train_interactions"]
    n_head = int(np.ceil(PROTOCOL["head_item_share"] * len(item_ids)))
    order = np.lexsort((np.asarray(item_ids), -popularity))
    head = np.zeros(len(item_ids), dtype=bool)
    head[order[:n_head]] = True
    return Protocol(user_ids, item_ids, targets, seen, popularity, tail, head, int(len(eligible)))


def leakage_report(train, valid, item_ids, items, protocol: Protocol) -> dict:
    """Checks that the pipeline respects the CP3 leakage rules. Raises on violation."""
    candidate_set = set(item_ids)
    train_pairs = set(zip(train.user_id, train.item_id))
    report = {
        "train_rows": int(len(train)),
        "validation_rows": int(len(valid)),
        "train_users": int(train.user_id.nunique()),
        "train_items": int(train.item_id.nunique()),
        "validation_users": int(valid.user_id.nunique()),
        "validation_users_seen_in_train": int(valid.user_id.drop_duplicates().isin(set(train.user_id)).sum()),
        "validation_rows_relevant": int((valid.rating >= PROTOCOL["relevant_rating_min"]).sum()),
        "validation_rows_relevant_warm_items": int(((valid.rating >= PROTOCOL["relevant_rating_min"])
                                                     & valid.item_id.isin(candidate_set)).sum()),
        "evaluable_users": len(protocol.user_ids),
        "evaluable_relevant_rows": protocol.relevant_rows,
        "candidate_items": len(item_ids),
        "candidate_ids_unique": len(candidate_set) == len(item_ids),
        "candidates_subset_of_train_items": candidate_set <= set(train.item_id),
        "candidates_with_metadata": int(pd.Series(item_ids).isin(set(items.item_id)).sum()),
        "train_max_timestamp": int(train.timestamp.max()),
        "validation_min_timestamp": int(valid.timestamp.min()),
        "temporal_order_respected": bool(train.timestamp.max() < valid.timestamp.min()),
        "train_validation_pair_overlap": int(len(train_pairs & set(zip(valid.user_id, valid.item_id)))),
        "long_tail_candidates": int(protocol.tail.sum()),
        "head_candidates": int(protocol.head.sum()),
        "relevant_long_tail_targets": int(sum(sum(bool(protocol.tail[i]) for i in t) for t in protocol.targets)),
        "item_feature_fields": ALLOWED_ITEM_FIELDS,
        "test_split_used": False,
    }
    assert report["candidate_ids_unique"], "candidate ids must be unique"
    assert report["candidates_subset_of_train_items"], "warm candidates must appear in training"
    assert report["candidates_with_metadata"] == len(item_ids), "every candidate needs metadata"
    assert report["temporal_order_respected"], "validation must be strictly after training"
    assert report["train_validation_pair_overlap"] == 0, "train/validation user-item pairs overlap"
    return report


# ---------------------------------------------------------------- ranking ---

def top_k(scores: np.ndarray, seen: set[int], k: int, tie_rank: np.ndarray) -> np.ndarray:
    """Top-k indices by descending score, excluding seen items; ties broken by item_id order."""
    scores = scores.astype(np.float64, copy=True)
    if seen:
        scores[np.fromiter(seen, dtype=np.int64)] = -np.inf
    # Over-select then sort exactly by (-score, item_id) to keep tie-breaking deterministic.
    m = min(len(scores), k * 4)
    pool = np.argpartition(-scores, m - 1)[:m]
    threshold = scores[pool].min()
    pool = np.union1d(pool, np.flatnonzero(scores >= threshold))
    chosen = pool[np.lexsort((tie_rank[pool], -scores[pool]))][:k]
    return chosen


# ---------------------------------------------------------------- metrics ---

def recall_at_k(rec, target: set[int], k: int) -> float:
    if not target:
        return 0.0
    return len(set(rec[:k]) & target) / len(target)


def ndcg_at_k(rec, target: set[int], k: int) -> float:
    dcg = sum(1.0 / np.log2(rank + 2) for rank, item in enumerate(rec[:k]) if item in target)
    ideal = sum(1.0 / np.log2(rank + 2) for rank in range(min(k, len(target))))
    return float(dcg / ideal) if ideal else 0.0


def gini(counts: np.ndarray) -> float:
    """Gini index of exposure over the whole candidate catalog (0 = equal, 1 = concentrated)."""
    x = np.sort(np.asarray(counts, dtype=np.float64))
    n, total = len(x), x.sum()
    if n == 0 or total == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * total))


def evaluate(recs: list[np.ndarray], protocol: Protocol, k: int = PROTOCOL["top_k"]):
    """Returns (summary metrics, per-user metrics DataFrame) for one model run."""
    per_user, tail_hits, tail_targets = [], 0, 0
    for u, rec in enumerate(recs):
        target = protocol.targets[u]
        per_user.append({"user_id": protocol.user_ids[u],
                         f"recall@{k}": recall_at_k(rec, target, k),
                         f"ndcg@{k}": ndcg_at_k(rec, target, k)})
        tail_target = {i for i in target if protocol.tail[i]}
        tail_targets += len(tail_target)
        tail_hits += len(set(rec[:k]) & tail_target)
    flat = np.concatenate([r[:k] for r in recs])
    exposure = np.bincount(flat, minlength=len(protocol.item_ids))
    detail = pd.DataFrame(per_user)
    summary = {
        f"recall@{k}": float(detail[f"recall@{k}"].mean()),
        f"ndcg@{k}": float(detail[f"ndcg@{k}"].mean()),
        f"catalog_coverage@{k}": float((exposure > 0).mean()),
        f"catalog_items_recommended@{k}": int((exposure > 0).sum()),
        f"average_recommended_popularity@{k}": float(protocol.popularity[flat].mean()),
        f"long_tail_exposure@{k}": float(protocol.tail[flat].mean()),
        f"head_exposure@{k}": float(protocol.head[flat].mean()),
        f"exposure_gini@{k}": gini(exposure),
        f"long_tail_relevant_hits@{k}": int(tail_hits),
        f"long_tail_recall@{k}": float(tail_hits / tail_targets) if tail_targets else 0.0,
        "users_evaluated": len(recs),
        "users_with_k_recs": int(sum(len(r) == k for r in recs)),
        "users_with_seen_item_recommended": int(sum(bool(set(r.tolist()) & protocol.seen[u])
                                                    for u, r in enumerate(recs))),
    }
    assert summary["users_with_seen_item_recommended"] == 0, "training items leaked into recommendations"
    assert summary["users_with_k_recs"] == len(recs), "every evaluable user must get k recommendations"
    return summary, detail


def load_development_data():
    """Loads only train + validation + candidates + allowed item fields (never test)."""
    train, valid = load_split("train"), load_split("validation")
    item_ids = load_candidates()
    items = load_items()
    protocol = build_protocol(train, valid, item_ids)
    return train, valid, item_ids, items, protocol


def main(output):
    train, valid, item_ids, items, protocol = load_development_data()
    report = {"protocol": PROTOCOL, "checks": leakage_report(train, valid, item_ids, items, protocol)}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write the CP3 protocol and leakage report")
    parser.add_argument("--output", default="outputs/cp3/protocol_report.json")
    main(parser.parse_args().output)
