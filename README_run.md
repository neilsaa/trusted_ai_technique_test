# How to run

## Requirements

Python 3.10+. Install:

```bash
pip install pandas numpy scikit-learn sentence-transformers
# optional, only for the Claude tool-calling path of the assistant:
pip install anthropic
```

> **Environment note (this machine):** `sentence-transformers` had a TensorFlow/Keras
> conflict; the scripts force the PyTorch backend by setting `USE_TF=0` before import, so no
> action is needed. Embeddings run on CPU (~3 min, one time).

## Step 1 — verify the data

```bash
python scripts/verify_dataset.py
```

## Step 2 — build the plot embeddings (once, ~3 min, cached)

```bash
python build_embeddings.py        # writes embeddings.npy
```

## Step 3 — run the assistant (no API key needed)

```bash
# the three required behaviours + edge cases, scripted:
python assistant.py --demo

# ad-hoc:
python assistant.py --user 1   --ask "what should I watch tonight?"
python assistant.py --user 1   --ask "what do people like me think of Pulp Fiction?"
python assistant.py --user 999 --ask "what should I watch tonight?"   # declines (unknown user)
```

Sample output is saved in `transcripts/sample_session.txt`.

### Optional — route through Claude instead of the deterministic router

Needs the `anthropic` SDK and credentials (`ANTHROPIC_API_KEY`, or `ant auth login`).
Uses `claude-opus-4-8` with the engine functions exposed as tools; Claude only chooses the
tool and phrases the reply — it cannot invent films.

```bash
python assistant.py --user 1 --ask "what should I watch tonight?" --llm
```

## Step 4 — reproduce the evaluation (Part 2)

```bash
python evaluate.py                 # full suite, 150 sampled test users, seed 42
python evaluate.py --users 300     # larger sample
python evaluate.py --seed 7        # different sample
```

Prints the ranking table (vs baselines), the content ablation, rating-prediction accuracy
(3 CF variants + baselines), and guardrail precision/recall. Full results are also written
to `eval_results.json`. Runtime: ~2–4 min on CPU.

## Files

| File | What it is |
|---|---|
| `data_layer.py` | Loads CSVs, builds the rating matrix + per-movie stats |
| `build_embeddings.py` | Precomputes plot embeddings → `embeddings.npy` |
| `content_engine.py` | Engine 1 — content/taste (plain / weighted / clustered) |
| `cf_engine.py` | Engine 2 — collaborative filtering (item-item + user-user) |
| `ease.py` | EASE — closed-form item-item model (added after a research pass; strongest CF baseline) |
| `recommender.py` | Hybrid blend + quality prior + guardrails + the 3 behaviours |
| `assistant.py` | Deterministic CLI assistant (+ optional Claude tool-calling) |
| `evaluate.py` | Reproducible evaluation (Part 2) |
| `star_experiment.py` | EXP-9 — tests STAR's temporal-decay + recency-limiting ingredients |
| `REPORT.md` | Parts 2–5 |
| `PLAN.md`, `EXPERIMENTS.md` | Design + the open decisions/experiments behind the build |
| `transcripts/` | Saved sample session |

**No API key is required** for the system or the evaluation — only the optional `--llm`
assistant path uses one.
