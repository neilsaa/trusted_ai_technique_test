# TrustedAI — Film Recommender (technical assessment)

A hybrid film-recommendation assistant built on MovieLens-small (610 users, 5,135 movies
with plot summaries, 74k ratings). It personalises recommendations, answers "what do people
with taste like mine think of film X?", and declines when the data can't support an answer.

## Start here

| Document | What's in it |
|---|---|
| **[REPORT.md](REPORT.md)** | **The main deliverable** — evaluation (Part 2), engineering note (Part 3), client handover (Part 4), AI-use note (Part 5) |
| **[README_run.md](README_run.md)** | How to run the system and reproduce the evaluation |
| [TASK.md](TASK.md) | The original assessment brief |
| [PLAN.md](PLAN.md) · [EXPERIMENTS.md](EXPERIMENTS.md) · [JOURNEY.md](JOURNEY.md) | Design, open decisions, and the end-to-end process record |

## Headline result

- **Collaborative filtering works** — roughly **2× a most-popular baseline** on every ranking
  metric; a relevant film in the user's top-10 for **~48%** of users (EASE).
- **The content/plot engine does *not* improve top-N ranking** (confirmed three ways) — kept
  only as a labelled cold-start fallback. Reported honestly rather than hidden.
- **Rating prediction** ("what do people like me think of X?") beats baselines by **~15%**
  (MAE 0.68 stars), answering ~85% of queries and declining the rest.
- **Guardrails** (decline on unknown user / out-of-catalogue film): precision & recall 1.0.

## Quick run

```bash
pip install pandas numpy scikit-learn            # core: no GPU, no API key
python assistant.py --demo                       # the 3 behaviours + edge cases
python evaluate.py                               # reproduce the evaluation (Part 2)
```

`embeddings.npy` is committed, so the evaluation and assistant run without installing
`sentence-transformers`/`torch` (those are only needed to *regenerate* embeddings via
`build_embeddings.py`). An optional Claude conversational wrapper (`assistant.py --llm`)
needs an Anthropic API key; everything else does not. See **README_run.md** for detail.
