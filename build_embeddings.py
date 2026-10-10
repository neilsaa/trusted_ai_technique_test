"""Precompute plot embeddings for every movie and cache them to disk.

Run once:  python build_embeddings.py

Writes embeddings.npy (n_movies x dim, float32), row-aligned to the movie order in
data_layer.load_dataset(). Forces the PyTorch backend of transformers to avoid the
TensorFlow/Keras-3 import conflict in this environment.
"""

from __future__ import annotations

import os

# Must be set before importing sentence_transformers / transformers.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from pathlib import Path

import numpy as np

from data_layer import load_dataset

MODEL_NAME = "all-MiniLM-L6-v2"
OUT_PATH = Path(__file__).resolve().parent / "embeddings.npy"
# What text we embed. EXP-2 can swap this for plot-only vs plot+genres.
EMBED_FIELDS = ("plot",)


def build_text(row) -> str:
    parts = []
    for f in EMBED_FIELDS:
        val = row.get(f)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    return "\n".join(parts) if parts else str(row.get("title", ""))


def main() -> None:
    from sentence_transformers import SentenceTransformer

    ds = load_dataset()
    texts = [build_text(r) for _, r in ds.movies.iterrows()]
    print(f"Embedding {len(texts)} plots with {MODEL_NAME} (fields={EMBED_FIELDS})...")

    model = SentenceTransformer(MODEL_NAME)
    emb = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # unit vectors -> cosine == dot product
    ).astype(np.float32)

    np.save(OUT_PATH, emb)
    print(f"Saved {emb.shape} -> {OUT_PATH}")


if __name__ == "__main__":
    main()
