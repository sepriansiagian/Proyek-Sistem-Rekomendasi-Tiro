"""Parametrised CP3 models. All parameters are fitted from training data and allowed item metadata only."""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import svds

TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")


def popularity_scores(protocol) -> np.ndarray:
    """Fixed definition: number of training interactions per candidate item."""
    return protocol.popularity.astype(np.float64)


def train_mf(train, protocol, factors: int, seed: int, max_iter: int = 7000):
    """Truncated SVD of the explicit 1-5 training rating matrix (all training users)."""
    train_users = sorted(train.user_id.unique())
    user_index = {u: i for i, u in enumerate(train_users)}
    item_index = {item: i for i, item in enumerate(protocol.item_ids)}
    known = train[train.item_id.isin(item_index)]
    matrix = coo_matrix((known.rating.to_numpy(dtype=np.float32),
                         (known.user_id.map(user_index).to_numpy(), known.item_id.map(item_index).to_numpy())),
                        shape=(len(train_users), len(protocol.item_ids)), dtype=np.float32).tocsr()
    u, s, vt = svds(matrix, k=factors, which="LM", tol=1e-4, maxiter=max_iter, random_state=seed)
    order = np.argsort(s)[::-1]
    user_factors = (u[:, order] * s[order]).astype(np.float32)
    rows = np.fromiter((user_index[x] for x in protocol.user_ids), dtype=np.int64)
    return user_factors[rows], vt[order, :].T.astype(np.float32)


def _tokens(text: str, ngram_max: int) -> list[str]:
    words = TOKEN_RE.findall(text.lower())
    grams = list(words)
    for n in range(2, ngram_max + 1):
        grams += [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
    return grams


def tfidf_item_matrix(items: pd.DataFrame, item_ids, max_features: int, ngram_max: int, min_df: int):
    """Word TF-IDF (sublinear tf, smoothed idf, L2 rows) over title + content_text_basic."""
    meta = items.drop_duplicates("item_id").set_index("item_id").reindex(item_ids)
    text = (meta["title"].fillna("").astype(str) + " " + meta["content_text_basic"].fillna("").astype(str))
    docs, dfs = [], Counter()
    for doc in text:
        counts = Counter(_tokens(doc, ngram_max))
        docs.append(counts)
        dfs.update(counts.keys())
    terms = sorted(((t, df) for t, df in dfs.items() if df >= min_df), key=lambda x: (-x[1], x[0]))
    vocab = {t: j for j, (t, _) in enumerate(terms[:max_features])}
    idf = np.array([np.log((1 + len(docs)) / (1 + dfs[t])) + 1 for t in vocab], dtype=np.float32)
    rows, cols, vals = [], [], []
    for i, counts in enumerate(docs):
        sel = [(vocab[t], np.log1p(c) * idf[vocab[t]]) for t, c in counts.items() if t in vocab]
        norm = np.sqrt(sum(v * v for _, v in sel))
        if norm:
            for j, v in sel:
                rows.append(i); cols.append(j); vals.append(v / norm)
    matrix = csr_matrix((np.asarray(vals, dtype=np.float32), (rows, cols)),
                        shape=(len(item_ids), len(vocab)), dtype=np.float32)
    return matrix, len(vocab)


def user_content_profiles(train, protocol, item_matrix):
    """Rating-weighted mean of training-item TF-IDF vectors (weight = max(rating-3, 0.25))."""
    users = {u: i for i, u in enumerate(protocol.user_ids)}
    items = {iid: j for j, iid in enumerate(protocol.item_ids)}
    known = train[train.user_id.isin(users) & train.item_id.isin(items)]
    weights = np.maximum(known.rating.to_numpy(dtype=np.float32) - 3.0, 0.25)
    user_item = csr_matrix((weights, (known.user_id.map(users).to_numpy(), known.item_id.map(items).to_numpy())),
                           shape=(len(users), len(items)), dtype=np.float32)
    profiles = (user_item @ item_matrix).tocsr()
    norms = np.sqrt(np.asarray(profiles.multiply(profiles).sum(axis=1)).ravel())
    norms[norms == 0] = 1
    return profiles.multiply((1.0 / norms)[:, None]).tocsr()


def rerank_scores(mf_batch: np.ndarray, seen_batch, protocol, alpha: float, beta: float) -> np.ndarray:
    """Track C: zscore_u(MF) - alpha * normalised log popularity + beta * 1[long tail].

    The z-score uses the mean/std of MF scores over the user's available (unseen) candidates.
    """
    pop = protocol.popularity
    pop_norm = np.log1p(pop) / np.log1p(max(int(pop.max()), 1))
    out = np.empty_like(mf_batch, dtype=np.float64)
    for r, (scores, seen) in enumerate(zip(mf_batch, seen_batch)):
        available = np.ones(len(scores), dtype=bool)
        if seen:
            available[list(seen)] = False
        sigma = float(scores[available].std()) or 1.0
        out[r] = (scores - float(scores[available].mean())) / sigma - alpha * pop_norm + beta * protocol.tail
    return out
