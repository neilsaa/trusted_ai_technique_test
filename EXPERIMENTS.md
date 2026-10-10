# Design Decisions & Experiments — running notes

For each step: what we do, **where there is more than one valid way**, and the
**experiment** that decides it on *this* data (don't assume — measure). Each experiment
feeds directly into Part 2 (evaluation) and gives us defensible answers for the interview.

Legend:  🔧 = a knob/choice   🧪 = an experiment that resolves it   ✅ = sensible default to start with

---

## Phase A — Data & setup
Straightforward, no real choices. Load `movies_with_plots.csv` + `ratings.csv`.

---

## Phase B — Plot embeddings (content engine input)

🔧 **Which embedding model?**
- ✅ `all-MiniLM-L6-v2` (local, free, 384-dim, fast)
- bigger local model (`all-mpnet-base-v2`, 768-dim, slower, usually better)
- paid API embeddings (Voyage/OpenAI) — better, costs money + adds a key dependency

🧪 **EXP-1:** Only if time allows — compare MiniLM vs mpnet on final ranking metrics.
Default to MiniLM; upgrading is a documented "next change," not a priority.

🔧 **What text do we embed?** plot only / plot + genres / plot + title + genres.
🧪 **EXP-2:** compare ranking metrics with plot-only vs plot+genres. (Cheap, worth doing.)

---

## Phase C — Engine 1: content / taste profile

🔧 **"Liked" threshold** — which films count as the user's taste?
- ✅ rating ≥ user's **own mean** (personal, corrects harsh/generous raters)
- rating ≥ fixed 4.0 (simpler, but unfair to harsh raters)
- rating ≥ user mean + margin (stricter)

🔧 **How to weight each liked film in the taste vector?**
- plain average (every liked film equal) — the baseline
- ✅ rating-weighted (5★ pulls harder than 4★)
- mean-centered weight (films below user mean pull *away*)

🔧 **One taste centre or several (multi-taste)?**
- ✅ k clusters, k ∈ {1,2,3} chosen by silhouette, capped by history size
- single average (k=1) — the baseline
- nearest-liked-film instead of any centre

🔧 **How to weight the clusters?**
- ✅ `(#films in cluster) × (avg rating)` normalised
- by size only / equal weight

🧪 **EXP-3 (content ablation):** run the content engine in increasing complexity —
(a) plain average, (b) rating-weighted, (c) weighted + clustered — and compare ranking
metrics. **Keep each addition only if it moves the number.** This is the core "does our
cleverness earn its place" experiment.

---

## Phase D — Engine 2: collaborative filtering

### Item-item CF (for recommendations)
🔧 **Similarity metric between movie columns:** cosine / adjusted-cosine / Pearson.
🔧 **Neighbourhood size** (how many similar movies to aggregate): top-N, N = 10/20/50.
🧪 **EXP-4:** compare cosine vs adjusted-cosine for item-item on ranking metrics.

### User-user CF (for the "opinion" question)
🔧 **Similarity metric between user rows** — the Pearson debate:
- raw cosine (no centering) — fooled by generous/harsh raters
- ✅ mean-centered / Pearson (corrects rater bias — but assumes own-mean = neutral)
- Pearson **+ co-rating minimum + shrinkage** (handles sparse overlap)

🔧 **Minimum co-rated films** before two users are "comparable" (sparsity guard):
none / ≥3 / ✅ ≥5. Below this the correlation is basically noise.

🔧 **Shrinkage** — down-weight similarity when overlap is small? ✅ yes / no.

🔧 **How many neighbours** to average over: k = 10/20/50.

🔧 **Decline rule** — minimum neighbours who rated the target film before we answer at
all (else "not enough signal"): ✅ ≥5.

🧪 **EXP-5 (the CF-similarity experiment — the one from our discussion):**
predict held-out ratings with user-user CF under three variants —
(a) raw cosine, (b) Pearson, (c) Pearson + co-rating-min + shrinkage —
report MAE/RMSE **and coverage** for each. Keep the winner on *this* data; report it.
Expectation to test, not assume: (c) best accuracy, lower coverage.

---

## Phase E — Quality prior

🔧 **Shrinkage constant `m0`** in `(v·mean_m + m0·global_mean)/(v + m0)`:
controls how much we trust a movie's own average vs the global average.
🧪 **EXP-6:** small sweep of `m0` (e.g. 5/10/20); pick on validation, don't over-tune.

---

## Phase F — The blend

🔧 **Blend weights `α (content), β (cf), γ (quality)`** — the central knob.
- 50/50/0 naive start
- research says **CF usually deserves more weight** (Meta-Prod2Vec α*≈0.15 on embedding
  side) — so expect β > α, but **don't copy the constant** (dataset-specific)
- per-item **adaptive** weights (lean content when a movie has few ratings, CF when many)
  — the "next change," addresses Burke's uniform-value flaw

🧪 **EXP-7 (blend sweep):** grid over (α,β,γ) on a **validation split**, pick the best by
ranking metrics, then report final numbers on the **held-out test split** (don't tune on
test). Include content-only and CF-only as points on the grid (ablation).

🧪 **EXP-8 (adaptive vs fixed):** if time — compare fixed blend vs per-item adaptive α.
Otherwise document as Part 3's proposed next change + what evidence would confirm it.

---

## Phase G — Evaluation harness (decisions about *how we measure*)

🔧 **Train/test split:**
- random hold-out — ❌ leaks future info [LEAK]
- leave-one-out — ❌ also leaks time
- ✅ **temporal** (hide each user's most recent ratings) — avoids leakage

🔧 **What counts as a "hit"?** any held-out film / only held-out films rated ≥ user mean
(✅ — we care about predicting *liked* films, not just watched).

🔧 **Metrics:**
- ranking: ✅ Recall@10, NDCG@10, Hit-rate@10 (also report @5, @20)
- rating prediction: ✅ MAE, RMSE + **coverage**
- decline: ✅ precision/recall on a labelled in/out-of-catalogue + known/unknown set
- 🔧 optional: catalogue coverage & diversity (beyond pure accuracy) — add if time

🔧 **Sample size:** all 610 users vs a seeded sample (~100–150). ✅ sample, stated + seeded.

🔧 **Baselines to beat:** ✅ random, most-popular, genre-match, content-only, CF-only.

---

## Summary — experiments in priority order

| # | Experiment | Decides | Priority |
|---|---|---|---|
| EXP-7 | Blend weight sweep (+ ablation) | α,β,γ; does hybrid beat each engine alone | **must** |
| EXP-3 | Content complexity ablation | is clustering/weighting worth it | **must** |
| EXP-5 | CF similarity variants | raw vs Pearson vs Pearson+shrinkage | **high** |
| EXP-6 | Quality-prior shrinkage | m0 | med |
| EXP-4 | Item-item similarity metric | cosine vs adjusted-cosine | med |
| EXP-2 | Embed plot vs plot+genres | input text | med |
| EXP-8 | Adaptive vs fixed blend | the "next change" | if time / else Part 3 |
| EXP-1 | Embedding model size | MiniLM vs mpnet | low |

**Guiding rule:** every added piece of cleverness must earn its place by improving a
metric vs a simpler version. Anything that doesn't is reported as a finding, not kept.

---

## References
- [LEAK] Ji et al., *Data Leakage in Recommender Offline Evaluation*, ACM TOIS — use time-based splits.
- [MP2V] Vasile et al., *Meta-Prod2Vec*, RecSys 2016 — linear embedding+CF blend; CF weighted heavily.
- [B02] Burke 2002 — weighted hybrids assume uniform per-item value; motivates adaptive α.
</content>
