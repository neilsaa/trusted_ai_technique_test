"""Engine 1 — content-based taste recommender (uses plot embeddings only).

Builds a user's taste from the plot embeddings of films they rated highly, then scores
every unseen film by similarity to that taste. Three modes let EXP-3 measure whether the
added complexity earns its place:

  * "plain"     : unweighted mean of liked-film embeddings (the baseline)
  * "weighted"  : mean weighted by (rating - user_mean)      (strength of preference)
  * "clustered" : k taste centres (k via silhouette), each weighted by size x strength
                  (handles multi-taste users)

All embeddings are unit-normalised (see build_embeddings.py), so cosine similarity is a
dot product.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from data_layer import Dataset

MAX_K = 3          # most taste centres we will split a user into
MIN_PER_CLUSTER = 5  # need at least this many liked films per candidate cluster


@dataclass
class TasteProfile:
    centres: np.ndarray      # (k, dim) unit-normalised taste centres
    weights: np.ndarray      # (k,) cluster weights, sum to 1
    n_liked: int
    k: int
    mode: str


class ContentEngine:
    def __init__(self, ds: Dataset, embeddings: np.ndarray):
        assert embeddings.shape[0] == ds.n_movies, "embeddings must align with movie index"
        self.ds = ds
        self.emb = embeddings

    # ---- taste construction ------------------------------------------------
    def _liked(self, user_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (movie_idxs, weights) of films rated >= the user's own mean.

        Weight = rating - user_mean (strength of preference, always > 0 for liked set).
        """
        rated = self.ds.user_ratings(user_id)
        if not rated:
            return np.array([], dtype=int), np.array([])
        umean = float(np.mean(list(rated.values())))
        idxs, wts = [], []
        for midx, r in rated.items():
            if r >= umean:
                idxs.append(midx)
                wts.append(max(r - umean, 1e-3))  # keep strictly positive
        return np.array(idxs, dtype=int), np.array(wts, dtype=float)

    def _pick_k(self, X: np.ndarray) -> int:
        """Choose number of taste centres by a quick silhouette check."""
        n = len(X)
        max_k = min(MAX_K, n // MIN_PER_CLUSTER)
        if max_k < 2:
            return 1
        best_k, best_score = 1, -1.0
        for k in range(2, max_k + 1):
            labels = KMeans(n_clusters=k, n_init=5, random_state=0).fit_predict(X)
            if len(set(labels)) < k:
                continue
            score = silhouette_score(X, labels)
            if score > best_score:
                best_k, best_score = k, score
        # Only split if clustering is clearly better than a single centre.
        return best_k if best_score > 0.10 else 1

    def taste_profile(self, user_id: int, mode: str = "clustered") -> TasteProfile | None:
        idxs, wts = self._liked(user_id)
        if len(idxs) == 0:
            return None
        X = self.emb[idxs]

        if mode == "plain":
            centre = _unit(X.mean(axis=0))
            return TasteProfile(centre[None, :], np.array([1.0]), len(idxs), 1, mode)

        if mode == "weighted":
            centre = _unit(np.average(X, axis=0, weights=wts))
            return TasteProfile(centre[None, :], np.array([1.0]), len(idxs), 1, mode)

        if mode == "clustered":
            k = self._pick_k(X)
            if k == 1:
                centre = _unit(np.average(X, axis=0, weights=wts))
                return TasteProfile(centre[None, :], np.array([1.0]), len(idxs), 1, mode)
            labels = KMeans(n_clusters=k, n_init=5, random_state=0).fit_predict(X)
            centres, cweights = [], []
            for c in range(k):
                m = labels == c
                if not m.any():
                    continue
                centres.append(_unit(np.average(X[m], axis=0, weights=wts[m])))
                # cluster weight = size x mean strength
                cweights.append(m.sum() * wts[m].mean())
            cweights = np.array(cweights, dtype=float)
            cweights /= cweights.sum()
            return TasteProfile(np.vstack(centres), cweights, len(idxs), len(centres), mode)

        raise ValueError(f"unknown mode {mode!r}")

    # ---- scoring -----------------------------------------------------------
    def scores(self, user_id: int, mode: str = "clustered") -> np.ndarray | None:
        """Content score for every movie (movie_idx order). Seen films are NOT masked
        here; the blend layer handles exclusion so each engine stays composable."""
        prof = self.taste_profile(user_id, mode)
        if prof is None:
            return None
        # (k, dim) . (dim, n_movies) -> (k, n_movies); weighted sum over clusters.
        sims = prof.centres @ self.emb.T          # cosine, since both unit-norm
        return prof.weights @ sims                # (n_movies,)

    def recommend(self, user_id: int, k: int = 10, mode: str = "clustered"):
        s = self.scores(user_id, mode)
        if s is None:
            return None
        seen = set(self.ds.user_ratings(user_id))
        order = np.argsort(-s)
        out = []
        for midx in order:
            if midx in seen:
                continue
            row = self.ds.movies.iloc[midx]
            out.append((int(row["movieId"]), row["title"], int(row["year"]), float(s[midx])))
            if len(out) == k:
                break
        return out


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


if __name__ == "__main__":
    import numpy as np
    from data_layer import load_dataset

    ds = load_dataset()
    emb = np.load("embeddings.npy")
    eng = ContentEngine(ds, emb)

    for uid in [1, 42, 414]:
        prof = eng.taste_profile(uid, "clustered")
        print(f"\n=== user {uid}: {prof.n_liked} liked films, k={prof.k} taste centres, "
              f"weights={np.round(prof.weights, 2)} ===")
        # show a couple of their top-rated films for context
        rated = ds.user_ratings(uid)
        top = sorted(rated.items(), key=lambda kv: -kv[1])[:5]
        print("  top-rated:", [f"{ds.movies.iloc[m]['title']} ({r})" for m, r in top])
        print("  content top-10:")
        for mid, title, year, sc in eng.recommend(uid, 10, "clustered"):
            print(f"    {sc:.3f}  {title} ({year})")
