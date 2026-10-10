"""Data layer for the film recommender.

Loads the MovieLens-filtered CSVs and builds the two core structures every engine
needs:
  * a dense rating matrix (users x movies) plus id<->index maps
  * per-movie rating statistics (count, mean) for the quality prior and CF

Everything downstream (content engine, CF engine, evaluation) imports from here so
there is a single source of truth for ids, ordering, and the train/test split.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data" / "ml-latest-small-filtered"


@dataclass
class Dataset:
    """In-memory view of the catalogue and ratings.

    Movies are indexed 0..n_movies-1 in the row order of movies_with_plots.csv; that
    same order is what build_embeddings.py writes embeddings.npy in, so the i-th
    embedding row always matches self.movies.iloc[i]. Users are indexed 0..n_users-1
    in sorted userId order.
    """

    movies: pd.DataFrame          # movieId, title, year, genres, plot  (+ movie_idx)
    ratings: pd.DataFrame         # userId, movieId, rating, timestamp   (+ user_idx, movie_idx)
    R: np.ndarray                 # (n_users, n_movies) float32, 0.0 where unrated
    movie_ids: np.ndarray         # movie_idx -> movieId
    user_ids: np.ndarray          # user_idx  -> userId

    # ---- id <-> index maps -------------------------------------------------
    @cached_property
    def _movieid_to_idx(self) -> dict[int, int]:
        return {int(mid): i for i, mid in enumerate(self.movie_ids)}

    @cached_property
    def _userid_to_idx(self) -> dict[int, int]:
        return {int(uid): i for i, uid in enumerate(self.user_ids)}

    def movie_idx(self, movie_id: int) -> int | None:
        return self._movieid_to_idx.get(int(movie_id))

    def user_idx(self, user_id: int) -> int | None:
        return self._userid_to_idx.get(int(user_id))

    @property
    def n_users(self) -> int:
        return self.R.shape[0]

    @property
    def n_movies(self) -> int:
        return self.R.shape[1]

    # ---- per-movie statistics ---------------------------------------------
    @cached_property
    def movie_rating_count(self) -> np.ndarray:
        """Number of ratings per movie (movie_idx order)."""
        return (self.R > 0).sum(axis=0).astype(np.int64)

    @cached_property
    def movie_rating_mean(self) -> np.ndarray:
        """Mean rating per movie over observed ratings; 0.0 for unrated movies."""
        counts = self.movie_rating_count
        sums = self.R.sum(axis=0)
        means = np.zeros(self.n_movies, dtype=np.float64)
        nz = counts > 0
        means[nz] = sums[nz] / counts[nz]
        return means

    @cached_property
    def global_rating_mean(self) -> float:
        observed = self.R[self.R > 0]
        return float(observed.mean()) if observed.size else 0.0

    # ---- user helpers ------------------------------------------------------
    def user_ratings(self, user_id: int) -> dict[int, float]:
        """Map movie_idx -> rating for one user, or {} if the user is unknown."""
        uidx = self.user_idx(user_id)
        if uidx is None:
            return {}
        row = self.R[uidx]
        nz = np.nonzero(row)[0]
        return {int(j): float(row[j]) for j in nz}

    def user_mean(self, user_id: int) -> float | None:
        rated = self.user_ratings(user_id)
        if not rated:
            return None
        return float(np.mean(list(rated.values())))

    # ---- catalogue lookup --------------------------------------------------
    def find_movie(self, title: str) -> pd.Series | None:
        """Case-insensitive title lookup. Returns the movie row or None if the film
        is not in the catalogue (drives the decline path). Tries exact match first,
        then a contains-match; ambiguous contains-matches return the most-rated one.
        """
        t = title.strip().lower()
        titles = self.movies["title"].str.lower()
        exact = self.movies[titles == t]
        if len(exact):
            return exact.iloc[0]
        contains = self.movies[titles.str.contains(t, regex=False)]
        if len(contains) == 0:
            return None
        if len(contains) == 1:
            return contains.iloc[0]
        # disambiguate by popularity (rating count)
        counts = self.movie_rating_count
        best = max(contains.index, key=lambda i: counts[self.movies.at[i, "movie_idx"]])
        return self.movies.loc[best]


def load_dataset(data_dir: Path = DATA_DIR) -> Dataset:
    movies = pd.read_csv(data_dir / "movies_with_plots.csv")
    ratings = pd.read_csv(data_dir / "ratings.csv")

    # Keep only ratings whose movie has a plot entry (alignment with embeddings).
    movies = movies.reset_index(drop=True).copy()
    movies["movie_idx"] = np.arange(len(movies), dtype=np.int64)
    movieid_to_idx = dict(zip(movies["movieId"].astype(int), movies["movie_idx"]))

    ratings = ratings[ratings["movieId"].isin(movieid_to_idx)].copy()

    user_ids = np.sort(ratings["userId"].unique())
    userid_to_idx = {int(uid): i for i, uid in enumerate(user_ids)}

    ratings["user_idx"] = ratings["userId"].map(userid_to_idx).astype(np.int64)
    ratings["movie_idx"] = ratings["movieId"].map(movieid_to_idx).astype(np.int64)

    n_users, n_movies = len(user_ids), len(movies)
    R = np.zeros((n_users, n_movies), dtype=np.float32)
    R[ratings["user_idx"].to_numpy(), ratings["movie_idx"].to_numpy()] = ratings["rating"].to_numpy()

    return Dataset(
        movies=movies,
        ratings=ratings,
        R=R,
        movie_ids=movies["movieId"].astype(int).to_numpy(),
        user_ids=user_ids.astype(int),
    )


if __name__ == "__main__":
    ds = load_dataset()
    print(f"users={ds.n_users}  movies={ds.n_movies}  ratings={len(ds.ratings):,}")
    print(f"global mean rating = {ds.global_rating_mean:.3f}")
    print(f"movies with >=1 rating = {(ds.movie_rating_count > 0).sum()}")
    print(f"movies with <5 ratings = {(ds.movie_rating_count < 5).sum()}")
    # sanity: a known user
    uid = int(ds.user_ids[0])
    rated = ds.user_ratings(uid)
    print(f"user {uid}: {len(rated)} ratings, mean {ds.user_mean(uid):.2f}")
    # sanity: catalogue lookup
    for q in ["Toy Story", "Pulp Fiction", "This Film Does Not Exist"]:
        m = ds.find_movie(q)
        print(f"lookup {q!r}: {'MISS' if m is None else m['title'] + ' (' + str(int(m['year'])) + ')'}")
