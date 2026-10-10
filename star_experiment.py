"""EXP-9 — do STAR's two training-free ingredients help on this data?

STAR (Google, 2024) scores a candidate by summing item-item similarities to the user's
history, where each history item is weighted by its rating AND by a temporal decay
lambda^(recency_rank), optionally keeping only the few most recent items. We already have
the two similarity signals (semantic plot embeddings + collaborative co-occurrence); the
only things STAR adds over our blend are:

  * temporal decay   — recent ratings count more (lambda < 1)
  * recency limiting  — use only the last-N rated items

This script measures both against the no-decay / full-history version (lambda=1, L=inf),
on the same temporal split and test users as evaluate.py, so the comparison is apples to
apples. Run:  python star_experiment.py

Scoring (embeddings unit-norm, so cosine == dot product):
    w_i    = max(rating_i - user_train_mean, eps) * lambda^rank_i   (rank 0 = most recent)
    sem(c) = sum_i w_i * cos(emb_c, emb_i)        = emb_c . (sum_i w_i emb_i)
    col(c) = sum_i w_i * cooccur_cos(c, i)
    score  = a * minmax(sem) + (1-a) * minmax(col)
Compared across lambda in {1.0, 0.9, 0.7}, L in {inf, 50, 20, 10, 3}, a in {0.0, 0.5}.
"""

from __future__ import annotations

import numpy as np

from data_layer import load_dataset
from evaluate import KS, make_temporal_split, ranking_metrics
from recommender import _minmax

EPS = 1e-3


def user_history_sorted(ds, split, uid):
    """Train-only (movie_idx, rating, timestamp) for a user, most-recent first."""
    r = ds.ratings
    sub = r[r["userId"] == uid]
    # keep only ratings still in the training matrix (held-out ones were zeroed)
    train_R = split.train_ds.R[ds.user_idx(uid)]
    rows = []
    for midx, rating, ts in zip(sub["movie_idx"], sub["rating"], sub["timestamp"]):
        if train_R[int(midx)] > 0:
            rows.append((int(midx), float(rating), int(ts)))
    rows.sort(key=lambda x: -x[2])   # most recent first
    return rows


def star_scores(ds, emb, item_norms, R, uid, split, lam, L, a):
    hist = user_history_sorted(ds, split, uid)
    if not hist:
        return None, set()
    tmean = float(np.mean([r for _, r, _ in hist]))
    liked = [(m, r) for (m, r, _) in hist if r >= tmean]
    if not liked:
        liked = [(m, r) for (m, r, _) in hist]
    if L is not None:
        liked = liked[:L]

    # temporal + rating weights (rank 0 = most recent liked item)
    weights = np.array([max(r - tmean, EPS) * (lam ** rank)
                        for rank, (_, r) in enumerate(liked)])
    idxs = np.array([m for m, _ in liked])

    # semantic: weighted taste vector, then cosine to all items
    taste = (weights[:, None] * emb[idxs]).sum(axis=0)
    sem = emb @ taste                         # unit-norm emb -> cosine up to |taste|

    # collaborative: weighted sum of item-item co-occurrence cosine
    col = np.zeros(ds.n_movies)
    for w, midx in zip(weights, idxs):
        col += w * (R.T @ R[:, midx]) / (item_norms * item_norms[midx])

    if a == 0.0:
        s = _minmax(col)
    elif a == 1.0:
        s = _minmax(sem)
    else:
        s = a * _minmax(sem) + (1 - a) * _minmax(col)
    return s, set(int(m) for m, _, _ in hist)


def eval_config(ds, emb, item_norms, R, split, users, lam, L, a):
    agg = {f"{m}@{k}": [] for k in KS for m in ("recall", "hit", "ndcg")}
    for uid in users:
        s, seen = star_scores(ds, emb, item_norms, R, uid, split, lam, L, a)
        if s is None:
            continue
        ranked = [int(m) for m in np.argsort(-s) if m not in seen]
        mm = ranking_metrics(ranked, set(split.test_liked[uid]))
        for key, val in mm.items():
            agg[key].append(val)
    return {key: float(np.mean(v)) if v else 0.0 for key, v in agg.items()}


def main():
    ds = load_dataset()
    emb = np.load("embeddings.npy")
    split = make_temporal_split(ds, seed=42)
    R = split.train_ds.R
    item_norms = np.linalg.norm(R, axis=0) + 1e-9

    rng = np.random.default_rng(42)
    users = sorted(rng.permutation(split.eval_users)[:150].tolist())
    print(f"STAR ingredient test on {len(users)} test users (temporal split, seed 42)\n")

    # CF-side (a=0): does temporal decay + recency limiting help the signal that works?
    print("=== Collaborative side (a=0.0) — decay x history-limit ===")
    print(f"{'lambda':>7} {'L':>5} {'R@10':>8} {'NDCG@10':>9} {'Hit@10':>8}")
    configs = [(1.0, None), (0.9, None), (0.7, None),
               (1.0, 50), (0.9, 50), (1.0, 20), (0.9, 20), (1.0, 10), (0.7, 3)]
    for lam, L in configs:
        m = eval_config(ds, emb, item_norms, R, split, users, lam, L, a=0.0)
        tag = "  <- baseline (no decay, full history)" if (lam == 1.0 and L is None) else ""
        print(f"{lam:>7} {str(L):>5} {m['recall@10']:>8.4f} {m['ndcg@10']:>9.4f} "
              f"{m['hit@10']:>8.3f}{tag}")

    # Full STAR blend (a=0.5) with its paper-recommended decay
    print("\n=== STAR blend (a=0.5, semantic+collaborative) ===")
    print(f"{'lambda':>7} {'L':>5} {'R@10':>8} {'NDCG@10':>9} {'Hit@10':>8}")
    for lam, L in [(1.0, None), (0.7, None), (0.7, 20), (0.7, 3)]:
        m = eval_config(ds, emb, item_norms, R, split, users, lam, L, a=0.5)
        print(f"{lam:>7} {str(L):>5} {m['recall@10']:>8.4f} {m['ndcg@10']:>9.4f} "
              f"{m['hit@10']:>8.3f}")


if __name__ == "__main__":
    main()
