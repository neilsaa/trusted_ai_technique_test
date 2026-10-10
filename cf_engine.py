"""Engine 2 — collaborative filtering (uses the rating matrix only, no plots).

Two uses, as per the plan:

  * item-item CF  -> a score for every unseen movie, feeding the top-K blend
  * user-user CF  -> a predicted rating for one (user, movie), answering
                     "what do people with taste like mine think of X?"

The user-user path supports the EXP-5 variants (raw cosine vs mean-centred/Pearson vs
Pearson + co-rating minimum + shrinkage) so we can measure, not assume, which is best on
this sparse data. It returns an explicit abstain when too few similar users have rated
the target film (the decline path).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data_layer import Dataset

# user-user CF defaults
N_NEIGHBOURS = 40
MIN_CORATED = 5        # min movies two users must share to be comparable
SHRINK_BETA = 10.0     # significance shrinkage: sim *= n_corated / (n_corated + beta)
MIN_SUPPORT = 3        # min neighbours who rated the target movie, else abstain


@dataclass
class OpinionResult:
    can_answer: bool
    predicted: float | None = None
    n_support: int = 0          # neighbours who rated the target film
    neighbour_mean: float | None = None
    reason: str = ""


class CFEngine:
    def __init__(self, ds: Dataset):
        self.ds = ds
        self.R = ds.R                                  # (n_users, n_movies)
        self.mask = (self.R > 0).astype(np.float32)
        # per-user mean over observed ratings (for mean-centring)
        counts = self.mask.sum(axis=1)
        sums = self.R.sum(axis=1)
        self.user_mean = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)

    # ---- item-item CF (for ranking) ---------------------------------------
    def _item_norms(self):
        if not hasattr(self, "_inorm"):
            self._inorm = np.linalg.norm(self.R, axis=0) + 1e-9
        return self._inorm

    def item_scores(self, user_id: int) -> np.ndarray | None:
        """Score every movie by summed cosine-similarity to the user's liked films.

        Uses mean-centred liked set: weight each liked film by (rating - user_mean) so
        films the user merely tolerated don't pull recommendations. Computed on the fly
        (no full 5k x 5k matrix needed) by projecting liked columns against all columns.
        """
        uidx = self.ds.user_idx(user_id)
        if uidx is None:
            return None
        rated = self.ds.user_ratings(user_id)
        if not rated:
            return None
        umean = self.user_mean[uidx]
        liked = {m: r for m, r in rated.items() if r >= umean}
        if not liked:
            liked = rated
        norms = self._item_norms()
        scores = np.zeros(self.ds.n_movies, dtype=np.float64)
        for midx, r in liked.items():
            col = self.R[:, midx]                       # (n_users,)
            sims = (self.R.T @ col) / (norms * norms[midx])  # cosine to every movie
            scores += (r - umean) * sims
        return scores

    # ---- user-user CF (for the opinion question) --------------------------
    def _user_similarities(self, uidx: int, variant: str) -> np.ndarray:
        """Similarity of every user to `uidx`. variant in {raw, pearson, pearson_shrunk}."""
        R = self.R
        target_mask = self.mask[uidx]

        if variant == "raw":
            A = R.copy()
        else:  # mean-centre each user over their observed ratings
            A = R - self.user_mean[:, None] * self.mask
        a = A[uidx]

        # cosine between target vector and every user, over each pair's shared support
        dots = A @ a
        norms_all = np.linalg.norm(A, axis=1) + 1e-9
        sims = dots / (norms_all * (np.linalg.norm(a) + 1e-9))

        # number of co-rated films with the target user
        corated = self.mask @ target_mask
        sims[corated < MIN_CORATED] = 0.0

        if variant == "pearson_shrunk":
            sims = sims * (corated / (corated + SHRINK_BETA))

        sims[uidx] = 0.0
        return sims

    def similar_users_opinion(
        self, user_id: int, movie_id: int, variant: str = "pearson_shrunk"
    ) -> OpinionResult:
        uidx = self.ds.user_idx(user_id)
        if uidx is None:
            return OpinionResult(False, reason="unknown user")
        midx = self.ds.movie_idx(movie_id)
        if midx is None:
            return OpinionResult(False, reason="movie not in catalogue")

        sims = self._user_similarities(uidx, variant)
        raters = np.nonzero(self.mask[:, midx])[0]          # users who rated the film
        raters = raters[raters != uidx]
        if len(raters) == 0:
            return OpinionResult(False, reason="no one has rated this film")

        # top neighbours (by similarity) who actually rated the target film
        rater_sims = sims[raters]
        keep = rater_sims > 0
        raters, rater_sims = raters[keep], rater_sims[keep]
        if len(raters) < MIN_SUPPORT:
            return OpinionResult(
                False, n_support=len(raters),
                reason=f"only {len(raters)} taste-neighbours have rated it (need {MIN_SUPPORT})",
            )
        order = np.argsort(-rater_sims)[:N_NEIGHBOURS]
        raters, rater_sims = raters[order], rater_sims[order]

        ratings = self.R[raters, midx]
        # similarity-weighted average, de-biased by each neighbour's mean then re-added
        centred = ratings - self.user_mean[raters]
        pred = self.user_mean[uidx] + np.average(centred, weights=rater_sims)
        pred = float(np.clip(pred, 0.5, 5.0))
        return OpinionResult(
            True, predicted=pred, n_support=len(raters),
            neighbour_mean=float(ratings.mean()),
            reason=f"{len(raters)} taste-neighbours rated it (avg {ratings.mean():.2f})",
        )


if __name__ == "__main__":
    from data_layer import load_dataset

    ds = load_dataset()
    cf = CFEngine(ds)

    # item-item ranking for a user
    uid = 1
    s = cf.item_scores(uid)
    seen = set(ds.user_ratings(uid))
    order = [m for m in np.argsort(-s) if m not in seen][:10]
    print(f"=== user {uid}: item-item CF top-10 ===")
    for midx in order:
        row = ds.movies.iloc[midx]
        print(f"    {s[midx]:.2f}  {row['title']} ({int(row['year'])})")

    # opinion question across the three EXP-5 variants
    pf = ds.find_movie("Pulp Fiction")
    print(f"\n=== user {uid}: opinion on {pf['title']} ===")
    for v in ["raw", "pearson", "pearson_shrunk"]:
        res = cf.similar_users_opinion(uid, int(pf["movieId"]), v)
        print(f"  [{v:15}] can_answer={res.can_answer} pred={res.predicted} "
              f"support={res.n_support} :: {res.reason}")

    # a sparse film should trigger the decline path
    sparse_idx = int(np.argmin(ds.movie_rating_count + (ds.movie_rating_count == 0) * 1e9))
    sparse = ds.movies.iloc[sparse_idx]
    print(f"\n=== decline test: sparse film {sparse['title']!r} "
          f"({ds.movie_rating_count[sparse_idx]} ratings) ===")
    res = cf.similar_users_opinion(uid, int(sparse["movieId"]))
    print(f"  can_answer={res.can_answer} :: {res.reason}")
