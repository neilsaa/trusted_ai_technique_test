"""EASE — Embarrassingly Shallow Autoencoder for sparse data (Steck, WWW 2019).

A closed-form item-item model: a single ridge-regressed item-item weight matrix B with a
zeroed diagonal. No iterative training, no GPU — one matrix inverse. On data this size it
fits in seconds on CPU, and it is a well-known strong baseline that routinely beats
neighbourhood CF and many deep models for top-N ranking.

    B = (X^T X + lambda*I)^-1  normalised so diag(B)=0
    scores(user) = x_user @ B

We fit on the binary interaction matrix (rated / not rated) — the standard implicit-
feedback formulation — which is what top-N ranking cares about.
"""

from __future__ import annotations

import numpy as np


def fit_ease(X: np.ndarray, l2: float = 250.0) -> np.ndarray:
    """X: (n_users, n_items) binary matrix. Returns item-item weight matrix B."""
    X = X.astype(np.float64)
    G = X.T @ X                              # (n_items, n_items) gram
    diag = np.diag_indices(G.shape[0])
    G[diag] += l2                            # ridge regularisation
    P = np.linalg.inv(G)
    B = P / (-np.diag(P))                    # closed-form solution
    B[diag] = 0.0                            # zero diagonal (no self-recommendation)
    return B


def ease_scores(X_user_row: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Score every item for one user from their binary interaction row."""
    return X_user_row.astype(np.float64) @ B


if __name__ == "__main__":
    import numpy as np
    from data_layer import load_dataset

    ds = load_dataset()
    Xbin = (ds.R > 0).astype(np.float64)
    print(f"Fitting EASE on {Xbin.shape} (binary)...")
    B = fit_ease(Xbin, l2=250.0)
    print(f"B shape {B.shape}, nonzero diag max {np.abs(np.diag(B)).max():.2e}")

    uid = 1
    uidx = ds.user_idx(uid)
    s = ease_scores(Xbin[uidx], B)
    seen = set(ds.user_ratings(uid))
    order = [m for m in np.argsort(-s) if m not in seen][:10]
    print(f"\nEASE top-10 for user {uid}:")
    for midx in order:
        row = ds.movies.iloc[midx]
        print(f"    {s[midx]:.3f}  {row['title']} ({int(row['year'])})")
