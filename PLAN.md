# Implementation Plan — TrustedAI Film Recommendation PoC

*Working plan. Grounded in the dataset (`data/ml-latest-small-filtered/`) and in the
verified research pass on real-world recommender practice (citations at the end).*

---

## 0. Design in one paragraph

A **hybrid recommender** — the industry-standard combination of **collaborative
filtering** (what people with similar taste rated highly) and **content-based
filtering** (plot-embedding similarity to what *you* rated highly) — blended into one
score. This is Burke's "weighted hybrid," the canonical design [B02]. On top of the
deterministic engine sits a thin **Claude tool-calling agent** that turns natural
language into engine calls and narrates the result. The engine is where all the
recommending happens, because the engine is what we can *measure* (Part 2).

```
   natural language  ─►  Claude agent  ─►  deterministic engine (the "truth")  ─►  CSVs
     (Layer B)                                  (Layer A)
```

**Why hybrid, in one line we can defend:** the data has dense users (~121 ratings each)
but sparse items (~51% of movies have <5 ratings). Pure CF fails on the sparse half of
the catalogue; plots exist for every movie and fill that gap. Cold-start is *the*
textbook motivation for hybridization [B02][CS].

---

## 1. Data (where it is, what we use)

All in `data/ml-latest-small-filtered/`:

| File | Rows | Used for |
|---|---|---|
| `movies_with_plots.csv` | 5,135 | **Content engine** — plot → embedding. (`movieId,title,year,genres,plot`) |
| `ratings.csv` | 74,064 | **CF engine + all evaluation ground truth.** (`userId,movieId,rating,timestamp`) |
| `movies.csv` | 5,135 | Title-with-year display/lookup |
| `tags.csv` | 2,440 | Ignore for core (too sparse per brief); optional flavour only |
| `links.csv` | 5,135 | Ignore (no external calls) |

Scale: **610 users, 5,135 movies, 74k ratings**, timestamps present (→ enables temporal split).

---

## 2. Architecture — two layers

### Layer A — the engine (`recommender.py`, pure Python, no LLM)

Deterministic functions over the CSVs. Everything measurable lives here.

| Function | Input | Output |
|---|---|---|
| `get_user_profile(userId)` | userId | liked films, top genres, rating count |
| `recommend_for_user(userId, k)` | userId, k | ranked list `[(movieId, score, reason)]` |
| `similar_users_opinion(userId, title)` | userId, movie | predicted rating + neighbour evidence |
| `search_by_plot(query, k)` | text | films with closest plots |
| `get_movie(title)` | title | movie record, or **not-in-catalogue** |

### Layer B — the agent (`assistant.py`, Claude + tools)

Claude with the five functions above as tools. System prompt rules:
- May only recommend films returned by a tool (no invention / no hallucinated titles).
- **Must decline** when a tool returns empty (unknown user, movie not in catalogue).
- Narrates the reason ("because you rated *Goodfellas* 5★").

The split is deliberate: you cannot put an honest number on a chatbot, but you can on
the engine underneath it.

---

## 3. One-time precompute (cached to disk)

1. **Plot embeddings** — each plot → 384-dim vector via `sentence-transformers`
   (`all-MiniLM-L6-v2`): local, free, reproducible, no API key for the heavy part.
   Cache to `embeddings.npy` (compute once, ~5k vectors).
2. **Rating matrix `R`** — sparse 610 × 5,135 from `ratings.csv`.
3. **Per-movie stats** — count + mean rating (for the quality prior and CF).

---

## 4. Engine 1 — Content-based (plot embeddings)

**Goal:** "what KIND of film suits you." Works even on near-unrated movies.

1. **Liked set** — films the user rated **≥ their own mean** (personal threshold, not a
   global 4.0 — corrects harsh vs generous raters).
2. **Rating-weighted taste** — weight each liked film's embedding by `(rating − user_mean)`
   so a 5★ pulls harder than a 4★. *(Fixes "ignores how much they liked it.")*
3. **Multi-taste clustering** — k-means over the liked embeddings; pick **k ∈ {1,2,3}**
   by a **quick silhouette check**, capped by history size (few films → k=1).
   *(Fixes "multi-taste blur.")*
4. **Cluster weights** — `weight(c) = (#films in c) × (avg rating in c)`, normalised to
   sum to 1 — so a 45-film cluster dominates a 3-film one. *(The point raised in review:
   dominance must be preserved.)*
5. **Content score** for each unseen movie:
   `content_score(m) = Σ_c weight(c) · cosine(embedding(m), centre(c))`

> Baseline variant for ablation: **k=1, unweighted average** = the simplest content model.

---

## 5. Engine 2 — Collaborative filtering (ratings)

**Goal:** "what people like you actually enjoyed." Captures quality/taste plots can't.

- **For recommendations** — **item-item CF**: cosine similarity between movie
  rating-vectors; score an unseen movie by its similarity to the user's liked films.
- **For the opinion question** — **user-user CF**: nearest users by **Pearson
  correlation** over co-rated films (Pearson, not cosine — corrects per-user rating
  bias); predicted rating = similarity-weighted average of those neighbours' ratings of
  the target film. Return **"N neighbours rated it, avg X.X"**; **decline if N is too
  small** (sparse movie → low confidence).

> Research note: CF usually deserves *more* weight than content where it's available
> (Meta-Prod2Vec's best blend put only ~0.15 on the embedding side) [MP2V]. We won't
> copy that constant — it's dataset-specific — but we won't default to 50/50 either.

---

## 6. Quality prior (a nudge, not an engine)

Shrunk (Bayesian) average rating so a 5.0 from 2 people doesn't beat a 4.3 from 300:

```
quality(m) = (v·mean_m + m0·global_mean) / (v + m0)     # v = #ratings, m0 = shrinkage const
```

Injects "is it actually good" into the final order.

---

## 7. The blend (final score)

Normalise the three to 0–1, then:

```
final(m) = α·content(m) + β·cf(m) + γ·quality(m)
```

- α, β, γ chosen **once on a validation split** — not hand-tuned to look good (brief
  explicitly doesn't reward hyperparameter fiddling).
- **Filter** seen films, **sort** desc, **return top-K** with scores + reasons.
- **Cold-start fallback:** too few ratings to build a taste → return
  **popular-in-genre, clearly labelled as a fallback**, not pretended personalisation.

> **Known limitation (cite + carry to Part 3):** fixed α/β assume each technique's value
> is uniform across all movies — but CF is weaker on low-rater movies [B02]. The clean
> next change is **per-item adaptive weighting**: lean on content when a movie has few
> ratings, on CF when it has many.

---

## 8. The three required behaviours → engine mapping

| User need | Path | Output |
|---|---|---|
| "What should I watch tonight?" | Engines 1+2+quality, blended | top-K ranked, unseen, personalised |
| "What do people like me think of *Pulp Fiction*?" | Engine 2 (user-user CF) | predicted rating + neighbour evidence |
| Film not in catalogue / unknown user | guardrail, stop early | **decline** (+ labelled fallback where sensible) |

---

## 9. Evaluation — Part 2 (the 35-pt centrepiece)

Everything offline, reproducible, grounded in `ratings.csv`. **`evaluate.py`** runs it
all and prints a table; we state sample sizes and the random seed.

### 9a. Split — temporal, to avoid leakage
Per user, sort ratings by timestamp; hold out the **most recent N** as test, train on
the rest. **Time-based, not random or plain leave-one-out** — random/LOO leak future
ratings into training and inflate metrics [LEAK]. This is a decision we can cite.

### 9b. Ranking quality (behaviour 1)
Hide each test user's recent **liked** films; ask `recommend_for_user` to rank; measure:
- **Recall@10, NDCG@10, Hit-rate@10** (standard top-K metrics).
- Over a **stated sample** (~100–150 users, seeded).

### 9c. Baselines — so a number *means* something
Compare the hybrid against:
- **Random**, **Most-popular**, **Genre-match**, and **each engine alone**
  (content-only, CF-only) — the ablation that shows the blend earns its place.
- **If the hybrid doesn't beat most-popular, we report that plainly** — the brief
  rewards it.

### 9d. Rating-prediction quality (behaviour 2)
Hold out known ratings; predict via user-user CF; report **MAE / RMSE vs baselines**
(global mean, per-movie mean) **and coverage** (how often CF can even answer — expected
low on sparse movies; that gap is itself a finding).

### 9e. Decline behaviour (behaviour 3)
Hand-built labelled set: in- vs out-of-catalogue titles, known vs unknown users →
measure refusal **precision / recall**.

### 9f. What the evaluation *cannot* show (scores positively)
- Offline accuracy ≠ real user satisfaction; held-out ratings are biased toward what
  users already chose to watch (popularity/exposure bias).
- Small, dated catalogue (1903–2014, 5k films).
- No measure of diversity/serendipity beyond accuracy unless we add coverage/diversity.

---

## 10. Deliverables & file layout

```
Test/
├── recommender.py      # Layer A: engines 1+2 + blend + guardrails
├── assistant.py        # Layer B: Claude agent + CLI
├── evaluate.py         # Part 2, reproducible, prints the results table
├── embeddings.npy      # cached plot embeddings
├── transcripts/        # sample outputs: 3 required scenarios + edge cases
├── REPORT.md           # Parts 2–5
├── README_run.md       # how to run system + eval (+ API-key note)
└── PLAN.md             # this file
```

---

## 11. Build order (incremental, each step measurable)

1. **Setup** — venv, deps (`pandas numpy scikit-learn sentence-transformers anthropic`),
   run `verify_dataset.py`.
2. **Engine 1 baseline** (k=1 average) → eyeball top-10 for a real user.
3. **Engine 1 full** (weighted + clustered).
4. **Engine 2** (item-item + user-user CF).
5. **Blend + quality prior + guardrails.**
6. **`evaluate.py`** — baselines first, then the hybrid; this is where the time goes.
7. **Agent layer** + save transcripts.
8. **Write** Parts 2–5 using the real numbers.

**If time runs short** (per brief, cut Part 1 first): ship content-engine +
popularity baseline + a CLI, no agent loop — and say so. Protect the evaluation.

---

## 12. Decisions already made (defensible in interview)

- **Local embeddings** (sentence-transformers), not a paid embedding API → free,
  reproducible, grader can run it.
- **Temporal split**, not random → avoids leakage [LEAK].
- **Measure every addition** against a baseline; keep only what moves the metric.
  Anything that doesn't is reported as a finding, not hidden.
- **Guardrails are first-class** — the system is allowed to say "I don't know."

---

## References (verified in the research pass)

- **[B02]** Burke, *Hybrid Recommender Systems: Survey and Experiments*, UMUAI 2002 —
  seven hybridization methods; warns linear blends assume uniform per-item value.
- **[MP2V]** Vasile et al., *Meta-Prod2Vec*, RecSys 2016 — linear embedding+CF blend
  beats either alone; best weight heavily favoured CF (α*=0.15 on embedding side).
- **[STAR]** Google, *STAR*, 2024 — weighted average of semantic + collaborative
  similarity; degrades when either component is removed.
- **[CS]** Google ML Crash Course; IEEE Access user-cold-start review 2023 — cold-start
  as the core motivation for hybrids; standard mitigations.
- **[LEAK]** Ji et al., *Data Leakage in Recommender Offline Evaluation*, ACM TOIS —
  random/leave-one-out splits leak future info; use time-based splits.
</content>
</invoke>
