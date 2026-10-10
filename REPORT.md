# TrustedAI Film Recommender — Report

*Parts 2–5. The system (Part 1) is in `recommender.py` (engine) and `assistant.py`
(assistant); run instructions are in `README_run.md`; sample outputs in
`transcripts/sample_session.txt`.*

**Time spent:** ~4 hours. **AI assistance:** heavy — see Part 5.

---

## What was built (one paragraph for context)

A **hybrid recommender** with two engines blended into one score, wrapped in a thin
assistant. Engine 1 (content) builds a taste profile from the plot embeddings of films a
user rated highly (rating-weighted, optionally split into k taste clusters) and scores
unseen films by cosine similarity. Engine 2 (collaborative filtering) scores films by
what similar users rated highly (item-item), and answers "what do people like me think of
X?" via user-user Pearson similarity. A shrunk-average quality prior is added, and the
three are min-max normalised and blended. Guardrails decline on unknown users and
out-of-catalogue films, and fall back to popular-in-genre for cold-start users.

---

## Part 2 — Evaluation

Everything here is produced by `python evaluate.py` (reproducible, seed 42). All numbers
below are over a **temporal split on 150 sampled test users**; blend weights were tuned on
a **disjoint 75-user validation split**, so the headline numbers are not tuned on the data
they report.

### Method and why

- **Split: temporal, not random.** For each user I hold out their most recent ~20% of
  ratings and train on the older ones. Random or leave-one-out splits let future ratings
  leak into training and inflate offline metrics [Ji et al., ACM TOIS]; a time-based split
  matches how the system would actually be used (predict the future from the past).
- **Ground truth for ranking = held-out films the user rated ≥ their own training mean**
  (films they actually *liked*, not merely watched). 590 of 610 users are evaluable
  (≥8 training ratings and ≥1 held-out liked film).
- **Metrics.** Ranking: Recall@K, NDCG@K, Hit-rate@K (K=5/10/20). Rating prediction:
  MAE + coverage. Guardrails: precision/recall on a labelled case set.
- **Baselines**, so the numbers mean something: random, most-popular, genre-match, and
  each engine alone (content-only, CF-only).

### Result 1 — Ranking ("what should I watch tonight?")

Over 150 test users (EASE tuned on the validation split, λ=250):

| Method | Recall@10 | Recall@20 | NDCG@5 | NDCG@10 | Hit-rate@10 |
|---|---|---|---|---|---|
| random | 0.000 | 0.004 | 0.000 | 0.001 | 0.01 |
| most-popular | 0.052 | 0.081 | 0.056 | 0.063 | 0.28 |
| genre-match | 0.021 | 0.042 | 0.036 | 0.034 | 0.19 |
| content-only | 0.010 | 0.019 | 0.007 | 0.010 | 0.04 |
| neighbourhood CF | 0.112 | 0.164 | **0.136** | **0.135** | 0.46 |
| hybrid (best weights) | 0.104 | 0.161 | 0.135 | 0.133 | 0.44 |
| **EASE** (closed-form) | **0.114** | **0.167** | 0.116 | 0.126 | **0.48** |

**Read this honestly:**

1. **Collaborative filtering is the real result.** Both CF methods roughly **double**
   most-popular on every metric (e.g. NDCG@10 0.135/0.126 vs 0.063; Hit@10 0.46/0.48 vs
   0.28) and put a film the user genuinely liked in the top-10 for **46–48% of users**.
   That is a clear, meaningful lift over the naive baseline.
2. **EASE vs neighbourhood CF is a genuine trade-off, not a clean win** (EASE was added
   after a research pass — see Part 5). EASE — a closed-form item-item model that fits in
   seconds — **wins on recall and hit-rate** (Recall@10 0.114 vs 0.112, Recall@20 0.167 vs
   0.164, Hit@10 **0.48 vs 0.46**: it surfaces a relevant film for more users), but
   **loses on NDCG** (0.126 vs 0.135: neighbourhood CF orders the very top of the list more
   sharply). Neither dominates. Which to prefer depends on the product goal — for a "10
   things to watch tonight" shelf, hit-rate/recall arguably matter as much as top-slot
   ordering, which tilts toward EASE; if only the top 1–2 slots are shown, NDCG tilts toward
   neighbourhood CF.
3. **The hybrid does *not* beat plain CF** (NDCG@10 0.133 vs 0.135). The weight sweep
   independently concluded the same: the best blend put **zero weight on content** (content
   0.00, CF 0.75, quality 0.15). On this offline ranking task content does not help — and
   content-only barely beats random.
4. **Why content fails *this metric*** (not a bug — verified two ways): content ranks the
   held-out liked films at a **median position of ~2,250 out of ~5,100** candidates. It
   surfaces obscure films whose *plots* match the user's taste (e.g. for a Se7en/Usual-
   Suspects fan it returns Psycho III, Friday the 13th Part 3), not the popular films users
   actually watch next. This is also a *known, published* property — text-derived item
   embeddings capture semantics but not collaborative co-occurrence signal [LLM2Rec, 2025].
   Offline accuracy is measured against films the user *chose* to watch, which skews popular
   — so it structurally cannot reward content's strengths (see "cannot show").

### Result 2 — Content ablation (EXP-3)

Does the added content machinery earn its place? Measured by NDCG@10 and by the median
rank of held-out liked films (lower = better):

| Content mode | NDCG@10 | median rank of liked films |
|---|---|---|
| plain average | 0.0019 | 2249 |
| rating-weighted | 0.0096 | 2260 |
| weighted + clustered | 0.0096 | 2260 |

Rating-weighting helps content ~5× on NDCG (0.0019 → 0.0096). **Clustering adds nothing**
on this data — the silhouette check almost always chooses k=1 (even for a user with 960
liked films), i.e. MovieLens tastes here aren't cleanly multi-modal in plot-embedding
space. Both facts are kept as findings rather than hidden: clustering is dead weight here
and I would cut it before production.

### Result 3 — Rating prediction ("what do people like me think of X?", EXP-5)

Over 4,025 held-out (user, film) pairs, predicting the held-out rating:

| Method | MAE | Coverage |
|---|---|---|
| global-mean baseline | 0.870 | 1.00 |
| per-movie-mean baseline | 0.809 | 1.00 |
| user-user CF, raw cosine | 0.692 | 0.86 |
| user-user CF, Pearson | **0.685** | 0.85 |
| user-user CF, Pearson + shrinkage | 0.685 | 0.85 |

**User-user CF clearly beats both baselines** (MAE 0.685 vs 0.809 for per-movie mean — a
~15% error reduction). Pearson (mean-centring) marginally beats raw cosine, confirming the
rater-bias correction helps a little on this data; shrinkage makes no measurable
difference here. CF answers **~85% of queries** — it declines the other ~15% because too
few taste-neighbours have rated the target film, which is the correct behaviour on sparse
items, not a failure.

### Result 4 — Guardrails (decline behaviour)

On a labelled set of 9 cases (unknown user, out-of-catalogue titles, known user + real
films): **precision 1.00, recall 1.00.** Every case that should decline did, and no valid
request was wrongly refused. (One early false positive turned out to be *correct* — "The
Matrix" is genuinely absent from this filtered catalogue, and the system rightly declined
it.)

### Result 5 — STAR's training-free ingredients (EXP-9, `star_experiment.py`)

After the recent-techniques research pass I tested the two things STAR (Google, 2024) adds
over my blend — **temporal decay** (recent ratings weighted more, λ<1) and **recency-limited
history** (use only the last-N items) — on the same split and test users:

| Variant (collaborative signal) | Recall@10 | NDCG@10 | Hit@10 |
|---|---|---|---|
| baseline (λ=1, full history) | 0.112 | **0.135** | 0.46 |
| temporal decay λ=0.9 | 0.108 | 0.134 | **0.49** |
| temporal decay λ=0.7 | **0.119** | 0.132 | 0.47 |
| last-20 items only | 0.109 | 0.129 | 0.49 |
| last-3 items only (STAR's best for Amazon) | 0.111 | 0.119 | 0.39 |

Two findings, both kept:

- **Temporal decay gives a small recall/hit-rate gain at a small NDCG cost** (λ=0.7: Recall@10
  +6%, NDCG −2%) — the same recall↔NDCG trade-off shape as EASE. Mildly worth it if the
  product optimises coverage.
- **Recency-limiting does *not* transfer to this domain.** STAR's best config keeps only the
  3 most recent items; here that *hurts* (NDCG 0.119 vs 0.135). Movie taste is stable, so
  discarding history loses signal — unlike STAR's sequential e-commerce setting. This is a
  concrete reminder that a paper's hyperparameters don't port across domains; you have to
  re-tune. (Re-adding the semantic/content side at a=0.5 again lowered everything, a third
  independent confirmation that content doesn't help top-N ranking here.)

### What this evaluation *cannot* tell you

This is the most important section, and it is specific to these results:

- **Offline accuracy ≠ user satisfaction, and it is popularity-biased.** Ground truth is
  films users already chose to watch, which over-represents popular films. This is exactly
  why CF (which leans popular) scores well and content (which surfaces the long tail)
  scores ~0. It does **not** prove content is useless to a real user — only that it does
  not predict already-watched films. Content's plausible real value (surprising the user,
  covering the ~2,600 films CF can barely rank) is invisible to this metric. I would not
  conclude "drop content" from the offline numbers alone; I'd A/B test it.
- **No measure of diversity, novelty, or serendipity.** A system that recommends the 10
  most popular acclaimed films scores well here and would feel generic in practice.
- **Rating prediction is evaluated only where CF can answer** (~85%). The 15% it declines
  are the hardest, sparsest cases — the MAE does not describe them.
- **Small, dated catalogue** (5,135 films, 1903–2014) with well-known titles missing.
  Conclusions may not transfer to a full modern catalogue.
- **The assistant's prose is not evaluated.** I evaluate the engine's structured output
  against ground truth; I do not measure whether the generated sentences are faithful or
  helpful (an LLM-judge pass would be the next addition).

---

## Part 3 — Engineering note

### 1. Would I ship this to the client's users?

**Not the hybrid as a ranker — but a CF-first version, yes, as a PoC.** The evidence says:

- **Ship a CF ranker.** Both CF methods are real, measured improvements over popularity
  (~2× on every metric; a relevant film in the top-10 for 46–48% of users). That is a
  defensible "it works" claim. Between them it's a trade-off (below).
- **Which CF: EASE or neighbourhood?** I'd ship **EASE** as the default. It wins on recall
  and hit-rate (surfaces a relevant film for more users), fits in seconds with one
  hyperparameter, and scales far better than neighbourhood CF (it's a single precomputed
  item-item matrix, not a per-request scan of all users). Neighbourhood CF's edge is
  sharper top-slot NDCG — I'd keep it as a candidate for an A/B test, or ensemble the two,
  rather than assume EASE's offline recall win transfers online.
- **Do not ship the content blend as-is for ranking** — it does not improve offline accuracy
  and the weight sweep zeroes it out. Keep content only for the cold-start/long-tail path
  where CF has no signal, and label it.
- **Ship the "opinion" feature** — CF rating prediction beats baselines by ~15% and declines
  honestly when unsure.
- **The guardrails are production-ready** (perfect on the labelled set) and are the single
  most important thing for a client who has to trust the system.

So: a **CF-first PoC (EASE) with honest declines** is shippable for exploration. A polished
consumer launch needs the online validation below first.

### 2. The next change I would make

**Per-item adaptive weighting between EASE and content**, instead of a single global blend.
The research surfaced the exact flaw in a fixed linear blend: it assumes each technique's
value is uniform across all items [Burke 2002], but CF/EASE is specifically weak on
low-interaction films. The change: lean on EASE where a film has many interactions, and on
content embeddings only where EASE is thin (the long tail). A verified recent result
(STAR, 2024) also suggests two cheap, training-free ingredients I'm *not* yet using and
would try first: **temporal decay** on the user's history and **limiting to the few most
recent items** — both free on CPU.

- **What I predict it improves:** recommendations for the ~2,600 sparse films EASE/CF can't
  rank well — i.e. coverage and long-tail quality — without hurting the head.
- **What it costs / could break:** more complexity and a second thing to tune; risk of
  surfacing more obscure films that *lower* the offline accuracy metric even if they help
  real users. That tension is the point — which is why the evidence to trust it is not the
  offline number.
- **Evidence that would convince me it worked:** an **online A/B test** on coverage-weighted
  engagement (do users actually watch the long-tail films it now surfaces?), plus a guardrail
  that offline NDCG@10 on the head doesn't regress by more than a small margin. Offline alone
  can't validate this change — by construction it trades head-accuracy for tail-coverage.

**What I deliberately did *not* pursue** (from the same research pass, with reasons): LLM2Rec
and CoLLM need GPU fine-tuning — infeasible here. LLM-reranking of a CF candidate list is
*actively risky* — two 2025 papers show reranking a homogeneous CF top-N can *lower*
accuracy because the candidates are too similar; it only helps over a diversified pool and
needs a paid API. So the research's net effect was to add one cheap, high-value baseline
(EASE) and to talk me *out* of the flashier LLM options for this setting.

### 3. What changes between this and production

- **What I'd monitor, and at what threshold:**
  - *Decline rate* for the opinion feature. It's ~15% offline; alert if it climbs past,
    say, **25%** sustained — that means the catalogue/ratings went sparse or an upstream
    join broke.
  - *Recommendation coverage* — fraction of the catalogue ever recommended. If it collapses
    (e.g. the same 50 popular films to everyone), that's popularity-bias runaway; alert if
    the top-50 films account for **>50%** of all impressions.
  - *Empty-result / fallback rate* — fraction of requests hitting the cold-start popular
    fallback. A spike means new users aren't getting history attached.
  - *Latency* p95 per request type.
- **What breaks first at 100× users/catalogue:** the CF step. Item-item CF here recomputes
  cosine over the full rating matrix per request, and user-user CF scans all users — both
  are O(users × items) per call and will not hold at scale. The fix is the standard one:
  precompute item-item similarities offline (nightly), or move to matrix factorization
  (ALS/SVD) with an approximate-nearest-neighbour index for retrieval. The embedding matrix
  also grows linearly and would move to a vector index. *I'm stating the scaling shape, not
  a latency number, since I have no traffic figures.*
- **What would make me roll it back:** the decline/guardrail precision dropping below
  ~0.95 (the system starts answering when it shouldn't — i.e. recommending films not in the
  catalogue or "personalising" for users with no history), or a coverage collapse to near-
  pure popularity (at that point it's a bestseller list, not a personaliser, and cheaper to
  run as one).

---

## Part 4 — Client handover note

*(One page, written for the client — a business owner. Kept jargon-free. Also readable as
a standalone handover.)*

**What this does.** Your users tell us who they are (a user id), and the assistant suggests
films it thinks they'll like, learning from the films that user has rated before. It can
also answer "what would people with taste like mine think of this film?", and it will
honestly say "I don't know" rather than guess when it can't help.

**What we commit to, and how it's measured.** On a fair test — we hide the most recent
films each user rated highly, then check whether the system would have recommended them —
the system finds at least one film the user genuinely liked **in its top 10 for about 46%
of users**. That is roughly **double** what a simple "most popular films" list achieves.
For the "what would people like me think of this film?" feature, its predicted score is off
by **about 0.7 of a star on a 5-star scale, on average** — clearly better than guessing the
film's overall average. These numbers come from a test you can re-run yourself (below).

**What this does *not* do.**
- It does **not** cover every film ever made — only the ~5,100 films in your catalogue
  (which span 1903–2014 and are missing some well-known titles). If a film isn't in the
  catalogue, it will say so, not invent an answer.
- It does **not** personalise for a brand-new user with no rating history — it shows
  popular films instead, and tells the user that's what it's doing.
- It is **not** a guarantee any individual recommendation is one that user will love — it
  improves the odds across many users; it does not promise a perfect pick every time.
- It does **not** currently prove it adds *variety* — a separate check we'd recommend
  before launch.

**How you can check the claim yourself — without taking our word for it.** Everything is
reproducible. Run `python evaluate.py` and you'll see the same table we're quoting,
computed on your own data, with the number of users each figure is based on printed next to
it. You can change which users are tested and re-run it. The sample conversations in
`transcripts/` show exactly what a user would see, including the cases where it declines.

**What happens when it gets something wrong.** Two kinds of "wrong":
- *It can't help* (unknown user, film not in catalogue, too little data) → it says so
  plainly and, where sensible, offers a popular-films fallback. In our tests it got this
  right **100%** of the time. This is the safe failure, and the one we optimised for.
- *It recommends a film the user doesn't enjoy* → this will happen; no recommender is
  perfect. The cost is low (a skipped suggestion), there's no harm, and the system keeps
  improving as the user rates more films. We'd monitor this in production via whether users
  actually watch what's suggested.

---

## Part 5 — Note on AI use

I used an AI assistant heavily and managed it closely. Specifically:

- **Design debate (what it got right).** I used it to pressure-test the architecture before
  writing code — the rating-weighted taste vector, multi-taste clustering, and the
  cluster-weighting-by-size idea all came out of that back-and-forth. It correctly pushed me
  toward a hybrid given the data's sparsity.
- **Literature grounding (right, and I verified it).** I ran two AI research passes over how
  real recommenders work (one on foundations, one on recent 2024–2026 techniques), each
  adversarially fact-checking its own claims against primary sources. The useful, *kept*
  outputs: hybrid-with-weighted-blend is standard (Burke 2002); CF usually deserves more
  weight than content (Meta-Prod2Vec); use a temporal split to avoid leakage (Ji et al.);
  **EASE is a strong closed-form baseline worth adding**, and the flashier LLM-reranking
  options are risky/infeasible here. I acted on the EASE finding — implemented it, and it
  turned out to win on recall/hit-rate while losing on NDCG, a trade-off I report rather
  than cherry-pick. I treated the rest as pointers and only kept claims I could tie to a
  source.
- **Code generation (right, with correction).** It wrote most of the engine and evaluation
  scaffolding. I overrode several things: it initially leaned toward a 50/50 content/CF
  blend, which the evaluation then *disproved* — the sweep found content should get ~0
  weight, so I changed the defaults to match the evidence rather than the prior.
- **What it got wrong / what I caught.** (1) An early "decline" test labelled *The Matrix*
  as an in-catalogue film; it's actually absent from this filtered dataset, so the test was
  wrong, not the system — I fixed the test. (2) The content engine looked broken (scored 0
  everywhere); I didn't accept that at face value — I dug in and confirmed it's a genuine
  property of the data (content ranks liked films near the middle), which became a central
  finding rather than a hidden bug. (3) I added a median-rank metric myself when the AI's
  top-K-only ablation couldn't distinguish the content variants.

The thing I'd want credit for is not the code — it's refusing to let the hybrid's nice-
looking demo output override what the numbers said. The headline finding of this submission
("the clever hybrid doesn't beat plain CF for ranking") is the opposite of what I set out
to build, and I'm reporting it because it's true.
