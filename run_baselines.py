"""W6 baselines and popularity-aware reranking for the long-tail track.

All learned quantities are derived from interactions_train.csv and item
metadata. Validation is read only for ranking metrics and examples.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import svds

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "w6"
CONFIG = {
    "mf_method": "TruncatedSVD on sparse explicit user-item rating matrix",
    "mf_rating_values": "explicit 1-5 ratings",
    "popularity_definition": "training interaction count per item",
    "tfidf_text_fields": ["title", "content_text_basic"],
    "rerank_formula": "zscore(MF) - penalty * normalized_log_popularity + bonus * long_tail_indicator",
    "seed": 42,
    "relevant_rating_min": 4.0,
    "top_k": 10,
    "long_tail_max_train_interactions": 5,
    "mf_factors": 32,
    "mf_max_solver_iterations": 7000,
    "tfidf_min_document_frequency": 2,
    "tfidf_max_features": 100000,
    "rerank_popularity_penalty": 1.0,
    "rerank_long_tail_bonus": 6.0,
    "candidate_file": "candidate_items_warm.csv",
    "candidate_policy": "warm candidates minus each user's train items",
    "evaluation_policy": "known train users with at least one relevant warm validation target",
}
TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")


def load_data():
    train = pd.read_csv(DATA / "interactions_train.csv")
    valid = pd.read_csv(DATA / "interactions_validation.csv")
    candidates = pd.read_csv(DATA / "candidate_items_warm.csv")["item_id"].astype(str)
    items = pd.read_csv(DATA / "items.csv", usecols=["item_id", "title", "content_text_basic"])
    for frame in (train, valid, items):
        frame["item_id"] = frame["item_id"].astype(str)
    return train, valid, candidates, items


def build_eval(train, valid, candidates):
    candidate_set = set(candidates)
    relevant = valid[(valid.rating >= CONFIG["relevant_rating_min"])
                     & valid.item_id.isin(candidate_set)]
    user_ids = sorted(set(train.user_id) & set(relevant.user_id))
    user_index = {u: i for i, u in enumerate(user_ids)}
    item_ids = list(candidates)
    item_index = {item: i for i, item in enumerate(item_ids)}
    targets = [set() for _ in user_ids]
    eligible = relevant[relevant.user_id.isin(user_index)]
    for row in eligible.itertuples(index=False):
        targets[user_index[row.user_id]].add(item_index[row.item_id])
    seen = [set() for _ in user_ids]
    for row in train[train.user_id.isin(user_index)].itertuples(index=False):
        seen[user_index[row.user_id]].add(item_index[row.item_id])
    return user_ids, item_ids, targets, seen, eligible


def tfidf_item_matrix(items, item_ids):
    """Dependency-free word TF-IDF implementation using SciPy CSR."""
    metadata = items.drop_duplicates("item_id").set_index("item_id")
    docs, dfs = [], Counter()
    for iid in item_ids:
        row = metadata.loc[iid]
        title = "" if pd.isna(row["title"]) else str(row["title"])
        content = "" if pd.isna(row["content_text_basic"]) else str(row["content_text_basic"])
        counts = Counter(TOKEN_RE.findall(f"{title} {content}".lower()))
        docs.append(counts)
        dfs.update(counts.keys())
    terms = [(term, df) for term, df in dfs.items()
             if df >= CONFIG["tfidf_min_document_frequency"]]
    terms.sort(key=lambda x: (-x[1], x[0]))
    vocab = {term: j for j, (term, _) in enumerate(terms[:CONFIG["tfidf_max_features"]])}
    idf = np.zeros(len(vocab), dtype=np.float32)
    for term, j in vocab.items():
        idf[j] = np.log((1 + len(docs)) / (1 + dfs[term])) + 1
    rows, cols, vals = [], [], []
    for i, counts in enumerate(docs):
        selected = [(vocab[t], np.log1p(count) * idf[vocab[t]])
                    for t, count in counts.items() if t in vocab]
        norm = np.sqrt(sum(value * value for _, value in selected))
        if norm:
            for j, value in selected:
                rows.append(i); cols.append(j); vals.append(value / norm)
    matrix = csr_matrix((np.asarray(vals, dtype=np.float32), (rows, cols)),
                        shape=(len(item_ids), len(vocab)), dtype=np.float32)
    return matrix, len(vocab)


def build_user_content_profiles(train, user_ids, item_ids, item_matrix):
    ui, users = ({iid: j for j, iid in enumerate(item_ids)},
                 {u: i for i, u in enumerate(user_ids)})
    known = train[train.user_id.isin(users) & train.item_id.isin(ui)]
    rows, cols, vals = [], [], []
    for row in known.itertuples(index=False):
        rows.append(users[row.user_id]); cols.append(ui[row.item_id])
        vals.append(float(max(row.rating - 3.0, 0.25)))
    user_item = csr_matrix((vals, (rows, cols)), shape=(len(user_ids), len(item_ids)), dtype=np.float32)
    profiles = (user_item @ item_matrix).tocsr()
    norms = np.sqrt(np.asarray(profiles.multiply(profiles).sum(axis=1)).ravel())
    norms[norms == 0] = 1
    return profiles.multiply((1.0 / norms)[:, None]).tocsr()


def train_mf(train, user_ids, item_ids):
    train_user_ids = sorted(train.user_id.unique())
    user_index, item_index = ({u: i for i, u in enumerate(train_user_ids)},
                              {item: i for i, item in enumerate(item_ids)})
    known = train[train.item_id.isin(item_index)]
    rows, cols = known.user_id.map(user_index).to_numpy(), known.item_id.map(item_index).to_numpy()
    values = known.rating.to_numpy(dtype=np.float32)
    matrix = coo_matrix((values, (rows, cols)), shape=(len(train_user_ids), len(item_ids)), dtype=np.float32).tocsr()
    # Truncated SVD gives a low-rank factorization of the sparse rating matrix.
    u, singular, vt = svds(matrix, k=CONFIG["mf_factors"], which="LM", tol=1e-4,
                           maxiter=CONFIG["mf_max_solver_iterations"], random_state=CONFIG["seed"])
    order = np.argsort(singular)[::-1]
    all_user_factors = (u[:, order] * singular[order]).astype(np.float32)
    eval_user_rows = np.fromiter((user_index[user] for user in user_ids), dtype=np.int64)
    return all_user_factors[eval_user_rows], vt[order, :].T.astype(np.float32)


def top_indices(scores, seen, k, tie_ids, tie_order):
    if seen:
        scores[np.fromiter(seen, dtype=np.int64)] = -np.inf
    threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
    above = np.flatnonzero(scores > threshold)
    needed = k - len(above)
    equal_in_id_order = tie_order[scores[tie_order] == threshold]
    chosen = np.concatenate((above, equal_in_id_order[:needed]))
    return chosen[np.lexsort((tie_ids[chosen], -scores[chosen]))]


def recommend_all(user_ids, item_ids, targets, seen, popularity, mf_u, mf_v, profiles, item_tfidf):
    k = CONFIG["top_k"]
    tail = popularity <= CONFIG["long_tail_max_train_interactions"]
    tie_ids = np.asarray(item_ids, dtype=str)
    tie_order = np.argsort(tie_ids)
    all_recs = {name: [] for name in ("popularity", "mf", "tfidf", "mf_reranked")}
    rows_metrics = []
    popularity_norm = np.log1p(popularity) / np.log1p(max(int(popularity.max()), 1))
    for u in range(len(user_ids)):
        mf_score = mf_u[u] @ mf_v.T
        cbf_score = (profiles.getrow(u) @ item_tfidf.T).toarray().ravel()
        choices = {
            "popularity": top_indices(popularity.astype(np.float32).copy(), seen[u], k, tie_ids, tie_order),
            "mf": top_indices(mf_score.copy(), seen[u], k, tie_ids, tie_order),
            "tfidf": top_indices(cbf_score.copy(), seen[u], k, tie_ids, tie_order),
        }
        available = np.ones(len(item_ids), dtype=bool)
        if seen[u]:
            available[list(seen[u])] = False
        sigma = float(mf_score[available].std()) or 1.0
        reranked = ((mf_score - float(mf_score[available].mean())) / sigma
                    - CONFIG["rerank_popularity_penalty"] * popularity_norm
                    + CONFIG["rerank_long_tail_bonus"] * tail)
        choices["mf_reranked"] = top_indices(reranked, seen[u], k, tie_ids, tie_order)
        for model, rec in choices.items():
            all_recs[model].append(rec)
            target = targets[u]
            dcg = sum(1 / np.log2(rank + 2) for rank, item in enumerate(rec) if item in target)
            ideal = sum(1 / np.log2(rank + 2) for rank in range(min(k, len(target))))
            rows_metrics.append({"model": model, "user_id": user_ids[u],
                                 "ndcg_at_k": float(dcg / ideal) if ideal else 0.0})
    return all_recs, rows_metrics, tail


def summarize(all_recs, metric_rows, item_ids, popularity, tail):
    k, detail, summaries = CONFIG["top_k"], pd.DataFrame(metric_rows), {}
    for model, recommendations in all_recs.items():
        flat = np.concatenate(recommendations)
        unique = np.unique(flat)
        summaries[model] = {
            f"ndcg@{k}": float(detail.loc[detail.model == model, "ndcg_at_k"].mean()),
            f"catalog_coverage@{k}": float(len(unique) / len(item_ids)),
            f"catalog_items_recommended@{k}": int(len(unique)),
            f"average_item_popularity@{k}": float(popularity[flat].mean()),
            f"long_tail_exposure@{k}": float(tail[flat].mean()),
            "users_evaluated": int(len(recommendations)),
        }
    return summaries


def main():
    np.random.seed(CONFIG["seed"])
    OUT.mkdir(parents=True, exist_ok=True)
    train, valid, candidates, items = load_data()
    user_ids, item_ids, targets, seen, eligible = build_eval(train, valid, candidates)
    popularity = train.groupby("item_id").size().reindex(item_ids).fillna(0).to_numpy(dtype=np.int32)
    print(f"Evaluation users: {len(user_ids):,}; relevant warm validation rows: {len(eligible):,}", flush=True)
    mf_u, mf_v = train_mf(train, user_ids, item_ids)
    print("MF trained", flush=True)
    item_tfidf, vocab_size = tfidf_item_matrix(items, item_ids)
    print(f"TF-IDF built ({vocab_size:,} terms)", flush=True)
    profiles = build_user_content_profiles(train, user_ids, item_ids, item_tfidf)
    all_recs, metric_rows, tail = recommend_all(
        user_ids, item_ids, targets, seen, popularity, mf_u, mf_v, profiles, item_tfidf)
    metrics = summarize(all_recs, metric_rows, item_ids, popularity, tail)
    config = dict(CONFIG, validation_evaluation_users=len(user_ids),
                  validation_relevant_warm_rows=int(len(eligible)), candidate_items=len(item_ids),
                  mf_training_users=int(train.user_id.nunique()),
                  mf_training_interactions=int(len(train)),
                  mf_training_unique_user_item_pairs=int(train.drop_duplicates(["user_id", "item_id"]).shape[0]),
                  tfidf_vocabulary_size=vocab_size,
                  outputs_use="validation sanity check; use a separate test split for final claims")
    (OUT / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pd.DataFrame(metric_rows).to_csv(OUT / "per_user_ndcg.csv", index=False)
    examples = []
    for model, rec_lists in all_recs.items():
        for user_idx in range(min(10, len(user_ids))):
            for rank, item_idx in enumerate(rec_lists[user_idx], start=1):
                examples.append({"model": model, "user_id": user_ids[user_idx], "rank": rank,
                                 "item_id": item_ids[item_idx], "train_popularity": int(popularity[item_idx]),
                                 "is_long_tail": bool(tail[item_idx]),
                                 "validation_relevant": item_idx in targets[user_idx]})
    pd.DataFrame(examples).to_csv(OUT / "top_n_examples.csv", index=False)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Saved experiment artifacts to {OUT}", flush=True)


if __name__ == "__main__":
    main()
