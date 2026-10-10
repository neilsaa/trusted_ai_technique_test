"""The hybrid recommender — Layer A, the deterministic engine behind everything.

Ties Engine 1 (content), Engine 2 (collaborative filtering) and a quality prior into one
blended score, applies guardrails (unknown user, out-of-catalogue, cold-start fallback),
and exposes the functions the agent layer and the evaluation both call:

    rec.recommend_for_user(user_id, k)        -> top-K personalised, or a fallback/decline
    rec.similar_users_opinion(user_id, title) -> predicted rating + evidence, or a decline
    rec.get_movie(title)                       -> catalogue lookup / not-in-catalogue
    rec.get_user_profile(user_id)              -> taste summary for narration

Scores from each engine are min-max normalised to [0,1] before blending so no single
component dominates by scale. Blend weights (alpha, beta, gamma) default to CF-leaning,
per the research, and are overridable so the evaluation can tune them on a validation
split.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cf_engine import CFEngine, OpinionResult
from content_engine import ContentEngine
from data_layer import Dataset, load_dataset

# Minimum ratings before we trust a personalised taste; below this -> cold-start fallback.
COLD_START_MIN = 5


@dataclass
class RecItem:
    movie_id: int
    title: str
    year: int
    score: float
    reason: str = ""


@dataclass
class RecResult:
    ok: bool
    items: list[RecItem] = field(default_factory=list)
    mode: str = ""          # "personalised" | "fallback:popular" | "decline"
    reason: str = ""


class HybridRecommender:
    def __init__(
        self,
        ds: Dataset,
        embeddings: np.ndarray,
        # Defaults are CF-dominant. The EXP-7 sweep found the offline-optimal content
        # weight is ~0 (content doesn't improve top-N accuracy on this data — see
        # eval_results.json / REPORT.md). We retain a SMALL content weight deliberately:
        # offline accuracy is popularity-biased and cannot credit content's cold-start
        # coverage and diversity. Per-item adaptive weighting is the principled fix (Part 3).
        alpha: float = 0.15,   # content weight (kept small, for cold-start/diversity)
        beta: float = 0.70,    # CF weight  (the empirically dominant signal)
        gamma: float = 0.15,   # quality-prior weight
        quality_shrink: float = 10.0,
        content_mode: str = "clustered",
    ):
        self.ds = ds
        self.content = ContentEngine(ds, embeddings)
        self.cf = CFEngine(ds)
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.content_mode = content_mode
        self.quality = self._quality_prior(quality_shrink)

    # ---- quality prior (shrunk average rating) ----------------------------
    def _quality_prior(self, m0: float) -> np.ndarray:
        v = self.ds.movie_rating_count.astype(float)
        mean = self.ds.movie_rating_mean
        gmean = self.ds.global_rating_mean
        return (v * mean + m0 * gmean) / (v + m0)

    # ---- guardrails / lookups ---------------------------------------------
    def get_movie(self, title: str):
        return self.ds.find_movie(title)

    def get_user_profile(self, user_id: int) -> dict | None:
        rated = self.ds.user_ratings(user_id)
        if not rated:
            return None
        umean = self.ds.user_mean(user_id)
        top = sorted(rated.items(), key=lambda kv: -kv[1])[:5]
        genres = {}
        for midx, r in rated.items():
            if r >= umean:
                for g in str(self.ds.movies.iloc[midx]["genres"]).split("|"):
                    genres[g] = genres.get(g, 0) + 1
        top_genres = sorted(genres, key=genres.get, reverse=True)[:5]
        return {
            "user_id": user_id,
            "n_ratings": len(rated),
            "mean_rating": round(umean, 2),
            "top_rated": [(self.ds.movies.iloc[m]["title"], r) for m, r in top],
            "top_genres": top_genres,
        }

    # ---- the main recommendation ------------------------------------------
    def recommend_for_user(self, user_id: int, k: int = 10) -> RecResult:
        if self.ds.user_idx(user_id) is None:
            return RecResult(False, mode="decline", reason=f"user {user_id} not found")

        rated = self.ds.user_ratings(user_id)
        if len(rated) < COLD_START_MIN:
            return self._popular_fallback(user_id, k, len(rated))

        content_s = self.content.scores(user_id, self.content_mode)
        cf_s = self.cf.item_scores(user_id)
        if content_s is None and cf_s is None:
            return self._popular_fallback(user_id, k, len(rated))

        parts, weights = [], []
        if content_s is not None:
            parts.append(_minmax(content_s)); weights.append(self.alpha)
        if cf_s is not None:
            parts.append(_minmax(cf_s)); weights.append(self.beta)
        parts.append(_minmax(self.quality)); weights.append(self.gamma)
        blended = np.average(np.vstack(parts), axis=0, weights=weights)

        seen = set(rated)
        order = np.argsort(-blended)
        items = []
        for midx in order:
            if midx in seen:
                continue
            row = self.ds.movies.iloc[midx]
            items.append(RecItem(
                int(row["movieId"]), row["title"], int(row["year"]),
                float(blended[midx]),
                reason=self._reason(midx, content_s, cf_s),
            ))
            if len(items) == k:
                break
        return RecResult(True, items=items, mode="personalised")

    def _reason(self, midx, content_s, cf_s) -> str:
        bits = []
        if cf_s is not None and _minmax(cf_s)[midx] > 0.6:
            bits.append("liked by users with similar taste")
        if content_s is not None and _minmax(content_s)[midx] > 0.6:
            bits.append("matches themes you rate highly")
        if self.quality[midx] >= 4.0:
            bits.append("highly rated overall")
        return "; ".join(bits) or "balanced match"

    def _popular_fallback(self, user_id, k, n_rated) -> RecResult:
        """Cold-start: not enough history -> popularity (shrunk avg), clearly labelled."""
        seen = set(self.ds.user_ratings(user_id))
        # require a reasonable number of ratings so "popular" means popular
        eligible = np.where(self.ds.movie_rating_count >= 20)[0]
        ranked = eligible[np.argsort(-self.quality[eligible])]
        items = []
        for midx in ranked:
            if midx in seen:
                continue
            row = self.ds.movies.iloc[midx]
            items.append(RecItem(int(row["movieId"]), row["title"], int(row["year"]),
                                 float(self.quality[midx]), reason="popular & highly rated"))
            if len(items) == k:
                break
        return RecResult(
            True, items=items, mode="fallback:popular",
            reason=f"user has only {n_rated} ratings (<{COLD_START_MIN}); "
                   "showing popular films rather than a personalised list",
        )

    # ---- opinion question (delegates to CF) -------------------------------
    def similar_users_opinion(self, user_id: int, title: str,
                              variant: str = "pearson_shrunk") -> dict:
        if self.ds.user_idx(user_id) is None:
            return {"ok": False, "reason": f"user {user_id} not found"}
        movie = self.ds.find_movie(title)
        if movie is None:
            return {"ok": False, "reason": f"'{title}' is not in the catalogue"}
        res: OpinionResult = self.cf.similar_users_opinion(
            user_id, int(movie["movieId"]), variant)
        return {
            "ok": res.can_answer,
            "title": movie["title"],
            "year": int(movie["year"]),
            "predicted_rating": res.predicted,
            "n_support": res.n_support,
            "neighbour_mean": res.neighbour_mean,
            "reason": res.reason,
        }


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def build_recommender(**kw) -> HybridRecommender:
    ds = load_dataset()
    emb = np.load("embeddings.npy")
    return HybridRecommender(ds, emb, **kw)


if __name__ == "__main__":
    rec = build_recommender()

    print("=== behaviour 1: what should I watch tonight? (user 1) ===")
    res = rec.recommend_for_user(1, 10)
    print(f"mode={res.mode}")
    for it in res.items:
        print(f"    {it.score:.3f}  {it.title} ({it.year})  — {it.reason}")

    print("\n=== behaviour 2: what do people like me think of Pulp Fiction? (user 1) ===")
    print("   ", rec.similar_users_opinion(1, "Pulp Fiction"))

    print("\n=== behaviour 3a: unknown user ===")
    print("   ", rec.recommend_for_user(99999, 5).__dict__)

    print("\n=== behaviour 3b: film not in catalogue ===")
    print("   ", rec.similar_users_opinion(1, "The Nonexistent Movie 2027"))
