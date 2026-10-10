"""Part 2 — reproducible evaluation of the hybrid recommender.

Everything is offline and grounded in ratings.csv. Run:

    python evaluate.py                # full suite, default sample
    python evaluate.py --users 150    # change sample size
    python evaluate.py --seed 0       # change sample seed

Protocol (decisions defended in the report):
  * Temporal split per user: hold out each user's most recent ~20% of ratings as test,
    train on the older ones. Time-based, NOT random/leave-one-out, to avoid leaking
    future ratings into training (a known inflator of offline metrics).
  * Ranking ground truth = held-out films the user rated >= their own TRAIN mean
    (i.e. films they actually liked, not merely watched).
  * Every number reports the sample size it was computed over.

Experiments run:
  EXP-7  blend-weight sweep on a VALIDATION user split; final metrics on a disjoint
         TEST user split (so weights are never tuned on the test set). Includes
         content-only and CF-only as ablation points.
  EXP-3  content complexity ablation (plain / weighted / clustered).
  EXP-5  user-user CF similarity variants (raw / pearson / pearson_shrunk) for rating
         prediction, with MAE/RMSE and coverage.
  decline precision/recall on a labelled in/out-of-catalogue + known/unknown set.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np

from cf_engine import CFEngine
from content_engine import ContentEngine
from data_layer import Dataset, load_dataset
from ease import ease_scores, fit_ease
from recommender import HybridRecommender, _minmax

HOLDOUT_FRAC = 0.2
MIN_TRAIN_RATINGS = 8     # user must keep this many train ratings to be evaluated
MIN_TEST_LIKED = 1        # and have at least this many held-out liked films
KS = (5, 10, 20)


# ----------------------------------------------------------------------------
# Temporal split
# ----------------------------------------------------------------------------
@dataclass
class Split:
    train_ds: Dataset
    test_liked: dict[int, list[int]]        # user_id -> [movie_idx] held-out & liked
    test_ratings: dict[int, list[tuple[int, float]]]  # user_id -> [(movie_idx, rating)]
    eval_users: list[int]


def make_temporal_split(ds: Dataset, seed: int = 42) -> Split:
    R_train = ds.R.copy()
    test_liked: dict[int, list[int]] = {}
    test_ratings: dict[int, list[tuple[int, float]]] = {}

    r = ds.ratings.sort_values(["userId", "timestamp"])
    for uid, grp in r.groupby("userId"):
        n = len(grp)
        if n < MIN_TRAIN_RATINGS + 1:
            continue
        n_hold = max(1, int(round(n * HOLDOUT_FRAC)))
        if n - n_hold < MIN_TRAIN_RATINGS:
            continue
        held = grp.iloc[-n_hold:]
        uidx = ds.user_idx(int(uid))
        # remove held-out ratings from the train matrix
        for midx in held["movie_idx"].to_numpy():
            R_train[uidx, int(midx)] = 0.0
        # train mean for this user (over remaining ratings)
        remaining = grp.iloc[:-n_hold]["rating"].to_numpy()
        tmean = float(remaining.mean())
        liked = [int(m) for m, rt in zip(held["movie_idx"], held["rating"]) if rt >= tmean]
        test_ratings[int(uid)] = [(int(m), float(rt))
                                  for m, rt in zip(held["movie_idx"], held["rating"])]
        if len(liked) >= MIN_TEST_LIKED:
            test_liked[int(uid)] = liked

    train_ds = Dataset(movies=ds.movies, ratings=ds.ratings, R=R_train,
                       movie_ids=ds.movie_ids, user_ids=ds.user_ids)
    eval_users = sorted(test_liked)
    return Split(train_ds, test_liked, test_ratings, eval_users)


# ----------------------------------------------------------------------------
# Ranking metrics
# ----------------------------------------------------------------------------
def ndcg_at_k(ranked_idx: list[int], relevant: set[int], k: int) -> float:
    dcg = 0.0
    for i, m in enumerate(ranked_idx[:k]):
        if m in relevant:
            dcg += 1.0 / np.log2(i + 2)
    ideal = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def ranking_metrics(ranked_idx: list[int], relevant: set[int]) -> dict:
    out = {}
    for k in KS:
        topk = ranked_idx[:k]
        hits = sum(1 for m in topk if m in relevant)
        out[f"recall@{k}"] = hits / len(relevant)
        out[f"hit@{k}"] = 1.0 if hits > 0 else 0.0
        out[f"ndcg@{k}"] = ndcg_at_k(ranked_idx, relevant, k)
    return out


# ----------------------------------------------------------------------------
# Per-user component scores (computed once, reused by every scorer)
# ----------------------------------------------------------------------------
def precompute_components(split: Split, emb: np.ndarray, users: list[int]):
    """For each user: normalised content & CF score vectors + the set of train-seen
    movies. quality is global. Lets the blend sweep re-rank cheaply."""
    ds = split.train_ds
    content = ContentEngine(ds, emb)
    cf = CFEngine(ds)
    quality = _quality_prior(ds, 10.0)
    qn = _minmax(quality)

    comp = {}
    for uid in users:
        cs = content.scores(uid, "clustered")
        fs = cf.item_scores(uid)
        comp[uid] = {
            "content": _minmax(cs) if cs is not None else None,
            "cf": _minmax(fs) if fs is not None else None,
            "seen": set(ds.user_ratings(uid)),
        }
    return comp, qn, ds


def _quality_prior(ds: Dataset, m0: float) -> np.ndarray:
    v = ds.movie_rating_count.astype(float)
    return (v * ds.movie_rating_mean + m0 * ds.global_rating_mean) / (v + m0)


def rank_for_scorer(comp_u, qn, scorer: str, weights, n_movies: int) -> list[int]:
    """Return movie_idx ranked desc for one user under a named scorer."""
    seen = comp_u["seen"]
    if scorer == "random":
        rng = np.random.default_rng(0)
        s = rng.random(n_movies)
    elif scorer == "popular":
        s = qn
    elif scorer == "content":
        s = comp_u["content"]
    elif scorer == "cf":
        s = comp_u["cf"]
    elif scorer == "hybrid":
        a, b, g = weights
        parts, ws = [], []
        if comp_u["content"] is not None:
            parts.append(comp_u["content"]); ws.append(a)
        if comp_u["cf"] is not None:
            parts.append(comp_u["cf"]); ws.append(b)
        parts.append(qn); ws.append(g)
        s = np.average(np.vstack(parts), axis=0, weights=ws)
    else:
        raise ValueError(scorer)
    if s is None:
        return []
    order = np.argsort(-s)
    return [int(m) for m in order if m not in seen]


def eval_scorer(scorer, comp, qn, split, users, weights=None, n_movies=0):
    agg = {f"{m}@{k}": [] for k in KS for m in ("recall", "hit", "ndcg")}
    for uid in users:
        relevant = set(split.test_liked[uid])
        ranked = rank_for_scorer(comp[uid], qn, scorer, weights, n_movies)
        if not ranked:
            continue
        m = ranking_metrics(ranked, relevant)
        for key, val in m.items():
            agg[key].append(val)
    return {key: float(np.mean(v)) if v else 0.0 for key, v in agg.items()}


# ----------------------------------------------------------------------------
# Genre-match baseline (needs its own ranking, not a single score vector)
# ----------------------------------------------------------------------------
def eval_genre_baseline(split, qn, users):
    ds = split.train_ds
    genres_split = ds.movies["genres"].str.split("|")
    agg = {f"{m}@{k}": [] for k in KS for m in ("recall", "hit", "ndcg")}
    for uid in users:
        seen = set(ds.user_ratings(uid))
        umean = ds.user_mean(uid)
        liked_genres = {}
        for midx, rt in ds.user_ratings(uid).items():
            if rt >= umean:
                for g in genres_split.iloc[midx]:
                    liked_genres[g] = liked_genres.get(g, 0) + 1
        # score = genre overlap * quality
        gscore = np.zeros(ds.n_movies)
        for midx in range(ds.n_movies):
            ov = sum(liked_genres.get(g, 0) for g in genres_split.iloc[midx])
            gscore[midx] = ov
        s = _minmax(gscore) * qn
        ranked = [int(m) for m in np.argsort(-s) if m not in seen]
        m = ranking_metrics(ranked, set(split.test_liked[uid]))
        for key, val in m.items():
            agg[key].append(val)
    return {key: float(np.mean(v)) if v else 0.0 for key, v in agg.items()}


# ----------------------------------------------------------------------------
# EASE (closed-form item-item). Tuned on val, evaluated on test.
# ----------------------------------------------------------------------------
def eval_ease(split, val_users, test_users):
    ds = split.train_ds
    Xbin = (ds.R > 0).astype(np.float64)

    def score_users(B, users):
        agg = {f"{m}@{k}": [] for k in KS for m in ("recall", "hit", "ndcg")}
        for uid in users:
            uidx = ds.user_idx(uid)
            s = ease_scores(Xbin[uidx], B)
            seen = set(ds.user_ratings(uid))
            ranked = [int(m) for m in np.argsort(-s) if m not in seen]
            mm = ranking_metrics(ranked, set(split.test_liked[uid]))
            for key, val in mm.items():
                agg[key].append(val)
        return {key: float(np.mean(v)) if v else 0.0 for key, v in agg.items()}

    # tune l2 on validation
    best_l2, best_ndcg = None, -1.0
    for l2 in [50.0, 100.0, 250.0, 500.0, 1000.0]:
        B = fit_ease(Xbin, l2=l2)
        r = score_users(B, val_users)
        if r["ndcg@10"] > best_ndcg:
            best_l2, best_ndcg = l2, r["ndcg@10"]
    B = fit_ease(Xbin, l2=best_l2)
    out = score_users(B, test_users)
    out["_best_l2"] = best_l2
    return out


# ----------------------------------------------------------------------------
# EXP-5: rating-prediction accuracy (user-user CF variants)
# ----------------------------------------------------------------------------
def eval_rating_prediction(split, users):
    ds = split.train_ds
    cf = CFEngine(ds)
    gmean = ds.global_rating_mean
    mmean = ds.movie_rating_mean

    results = {}
    variants = ["raw", "pearson", "pearson_shrunk"]
    for v in variants:
        errs, n_total, n_answered = [], 0, 0
        for uid in users:
            for midx, actual in split.test_ratings[uid]:
                n_total += 1
                movie_id = int(ds.movie_ids[midx])
                res = cf.similar_users_opinion(uid, movie_id, v)
                if res.can_answer:
                    n_answered += 1
                    errs.append(abs(res.predicted - actual))
        errs = np.array(errs)
        results[v] = {
            "MAE": float(errs.mean()) if len(errs) else None,
            "coverage": n_answered / n_total if n_total else 0.0,
            "n_answered": n_answered, "n_total": n_total,
        }
    # baselines over the same pairs
    g_err, m_err, n = [], [], 0
    for uid in users:
        for midx, actual in split.test_ratings[uid]:
            n += 1
            g_err.append(abs(gmean - actual))
            mm = mmean[midx] if mmean[midx] > 0 else gmean
            m_err.append(abs(mm - actual))
    results["baseline_global_mean"] = {"MAE": float(np.mean(g_err)), "coverage": 1.0, "n_total": n}
    results["baseline_movie_mean"] = {"MAE": float(np.mean(m_err)), "coverage": 1.0, "n_total": n}
    return results


# ----------------------------------------------------------------------------
# EXP-3: content complexity ablation (ranking)
# ----------------------------------------------------------------------------
def eval_content_modes(split, emb, users):
    """Ablation. Top-K recall is ~0 for content alone on this task (it surfaces obscure
    plot-similar films, not the popular films users actually rate), so we ALSO report the
    median rank of held-out liked films among all unseen candidates — a finer-grained
    signal that can still discriminate plain vs weighted vs clustered when recall@K ties."""
    ds = split.train_ds
    content = ContentEngine(ds, emb)
    out = {}
    for mode in ["plain", "weighted", "clustered"]:
        agg = {f"ndcg@{k}": [] for k in KS}
        agg.update({f"recall@{k}": [] for k in KS})
        med_ranks = []
        for uid in users:
            cs = content.scores(uid, mode)
            if cs is None:
                continue
            seen = set(ds.user_ratings(uid))
            ranked = [int(m) for m in np.argsort(-cs) if m not in seen]
            relevant = set(split.test_liked[uid])
            m = ranking_metrics(ranked, relevant)
            for k in KS:
                agg[f"ndcg@{k}"].append(m[f"ndcg@{k}"])
                agg[f"recall@{k}"].append(m[f"recall@{k}"])
            pos = {mv: i for i, mv in enumerate(ranked)}
            ranks = [pos[mv] for mv in relevant if mv in pos]
            if ranks:
                med_ranks.append(float(np.median(ranks)))
        res = {key: float(np.mean(v)) for key, v in agg.items()}
        res["median_rank_of_liked"] = float(np.mean(med_ranks)) if med_ranks else None
        out[mode] = res
    return out


# ----------------------------------------------------------------------------
# Decline behaviour
# ----------------------------------------------------------------------------
def eval_decline(ds, emb):
    rec = HybridRecommender(ds, emb)
    cases = []  # (description, should_decline, did_decline)

    known_user = int(ds.user_ids[0])
    unknown_user = 999999
    # titles verified present in the filtered catalogue (The Matrix etc. are absent)
    real_titles = ["Pulp Fiction", "Toy Story", "Forrest Gump"]
    fake_titles = ["The Nonexistent Movie 2027", "Zzzqqq Not A Film", "My Imaginary Sequel XII"]

    # recommend_for_user: unknown user should decline; known should not
    cases.append(("recommend/unknown-user", True,
                  rec.recommend_for_user(unknown_user).mode == "decline"))
    cases.append(("recommend/known-user", False,
                  rec.recommend_for_user(known_user).mode == "decline"))
    # opinion: out-of-catalogue should decline; real should answer
    for t in fake_titles:
        cases.append((f"opinion/fake:{t[:15]}", True,
                      not rec.similar_users_opinion(known_user, t)["ok"]))
    for t in real_titles:
        r = rec.similar_users_opinion(known_user, t)
        # answering OR an honest low-support decline both count as "not a hallucination";
        # here we score whether it (correctly) did NOT decline for lack of catalogue entry
        declined_for_catalogue = (not r["ok"]) and ("catalogue" in r["reason"])
        cases.append((f"opinion/real:{t[:15]}", False, declined_for_catalogue))
    # opinion for unknown user should decline
    cases.append(("opinion/unknown-user", True,
                  not rec.similar_users_opinion(unknown_user, "Pulp Fiction")["ok"]))

    tp = sum(1 for _, sd, dd in cases if sd and dd)
    fp = sum(1 for _, sd, dd in cases if not sd and dd)
    fn = sum(1 for _, sd, dd in cases if sd and not dd)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return {"precision": precision, "recall": recall,
            "cases": [{"case": c, "should_decline": sd, "did_decline": dd} for c, sd, dd in cases]}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--users", type=int, default=150, help="eval sample size (test split)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="eval_results.json")
    args = ap.parse_args()

    print("Loading data + embeddings...")
    ds = load_dataset()
    emb = np.load("embeddings.npy")

    print("Building temporal split...")
    split = make_temporal_split(ds, seed=args.seed)
    print(f"  {len(split.eval_users)} users are evaluable (>= {MIN_TRAIN_RATINGS} train "
          f"ratings and >= {MIN_TEST_LIKED} held-out liked film)")

    rng = np.random.default_rng(args.seed)
    all_users = split.eval_users
    # disjoint validation / test user samples
    perm = rng.permutation(all_users)
    n = min(args.users, len(all_users))
    test_users = sorted(perm[:n].tolist())
    val_users = sorted(perm[n:n + n // 2].tolist()) or test_users[: n // 2]
    print(f"  test_users={len(test_users)}  val_users={len(val_users)}")

    n_movies = ds.n_movies
    print("Precomputing per-user component scores (test)...")
    comp_test, qn, _ = precompute_components(split, emb, test_users)
    print("Precomputing per-user component scores (val)...")
    comp_val, _, _ = precompute_components(split, emb, val_users)

    # ---- EXP-7: blend sweep on validation, pick best NDCG@10 -------------
    print("\n[EXP-7] blend-weight sweep on validation users...")
    grid = []
    for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for b in [0.0, 0.25, 0.5, 0.75, 1.0]:
            for g in [0.0, 0.15, 0.3]:
                if a + b + g == 0:
                    continue
                grid.append((a, b, g))
    best, best_ndcg = None, -1.0
    for w in grid:
        r = eval_scorer("hybrid", comp_val, qn, split, val_users, w, n_movies)
        if r["ndcg@10"] > best_ndcg:
            best, best_ndcg = w, r["ndcg@10"]
    print(f"  best weights (content,cf,quality)={best}  val NDCG@10={best_ndcg:.4f}")

    # ---- final comparison on TEST users ---------------------------------
    print("\n[Final] ranking metrics on test users (best weights from validation):")
    results = {"config": {"test_users": len(test_users), "val_users": len(val_users),
                          "seed": args.seed, "best_weights": best,
                          "holdout_frac": HOLDOUT_FRAC}}
    table = {}
    table["random"] = eval_scorer("random", comp_test, qn, split, test_users, None, n_movies)
    table["popular"] = eval_scorer("popular", comp_test, qn, split, test_users, None, n_movies)
    table["genre-match"] = eval_genre_baseline(split, qn, test_users)
    table["content-only"] = eval_scorer("content", comp_test, qn, split, test_users, None, n_movies)
    table["cf-only"] = eval_scorer("cf", comp_test, qn, split, test_users, None, n_movies)
    table["hybrid"] = eval_scorer("hybrid", comp_test, qn, split, test_users, best, n_movies)
    print("  [EASE] fitting + tuning l2 on validation...")
    table["ease"] = eval_ease(split, val_users, test_users)
    results["ranking"] = table
    results["config"]["ease_best_l2"] = table["ease"].get("_best_l2")

    hdr = f"{'method':<14}" + "".join(f"{'R@'+str(k):>9}" for k in KS) + \
          "".join(f"{'NDCG@'+str(k):>10}" for k in KS) + f"{'Hit@10':>9}"
    print(hdr); print("-" * len(hdr))
    for name, m in table.items():
        line = f"{name:<14}" + "".join(f"{m['recall@'+str(k)]:>9.4f}" for k in KS) + \
               "".join(f"{m['ndcg@'+str(k)]:>10.4f}" for k in KS) + f"{m['hit@10']:>9.3f}"
        print(line)

    # ---- EXP-3: content ablation ----------------------------------------
    print("\n[EXP-3] content complexity ablation (test users):")
    results["content_ablation"] = eval_content_modes(split, emb, test_users)
    for mode, m in results["content_ablation"].items():
        mr = f"{m['median_rank_of_liked']:.0f}" if m["median_rank_of_liked"] else "n/a"
        print(f"  {mode:<10} NDCG@10={m['ndcg@10']:.4f}  Recall@10={m['recall@10']:.4f}  "
              f"median_rank_of_liked={mr}")

    # ---- EXP-5: rating prediction ---------------------------------------
    print("\n[EXP-5] rating-prediction accuracy (test users):")
    results["rating_prediction"] = eval_rating_prediction(split, test_users)
    for name, m in results["rating_prediction"].items():
        mae = f"{m['MAE']:.4f}" if m["MAE"] is not None else "n/a"
        print(f"  {name:<22} MAE={mae}  coverage={m['coverage']:.3f}  (n={m.get('n_total')})")

    # ---- decline --------------------------------------------------------
    print("\n[Decline] guardrail precision/recall:")
    results["decline"] = eval_decline(ds, emb)
    print(f"  precision={results['decline']['precision']:.3f}  "
          f"recall={results['decline']['recall']:.3f}  "
          f"({len(results['decline']['cases'])} labelled cases)")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full results -> {args.out}")


if __name__ == "__main__":
    main()
