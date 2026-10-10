# Project Journey — from brief to final result

A chronological record of what we did, what came out of each step, how and why we enhanced
it, where each idea came from, and what we ended with. Written as interview-prep: every
decision here is one you should be able to defend.

---

## Phase 0 — Understand the task and the data

**What we did.** Read the brief and inspected the dataset before writing any code.

**Output / what we learned:**
- The test scores *reasoning*, not system quality. Weights: Evaluation 35, Engineering
  note 25, Client handover 20, System 15, AI-use 5. Reporting an honest "no better than
  naive" result is explicitly a *good* outcome.
- The data: 610 users, 5,135 movies **with plot text**, 74,064 ratings. The decisive fact:
  **users are dense (~121 ratings each) but items are sparse (~51% of movies have <5
  ratings).**

**Why it mattered.** That sparsity is the single fact that drove the whole design: pure
collaborative filtering fails on the sparse half of the catalogue, and plots exist for
every film — so a **hybrid** (content + collaborative) is the natural answer. This wasn't a
default; it was a reasoned response to the data.

---

## Phase 1 — Design, shaped by a Q&A

**What we did.** Worked through the design conversationally, refining it under questioning
(these refinements are real and defensible because they came from spotting weaknesses):
- Represent a user's taste as the **rating-weighted average of the plot embeddings** of
  films they liked. → *Weakness spotted:* a plain average ignores how much they liked each
  film. *Fix:* weight by rating.
- → *Weakness spotted:* one average blurs a multi-taste user (horror + rom-com lands in the
  empty middle). *Fix:* **cluster** liked films into k taste centres.
- → *Weakness spotted:* equal clusters ignore dominance. *Fix:* **weight each cluster by
  size × strength**.
- → *Weakness spotted:* plot similarity ≠ quality. *Fix:* that's what the **CF engine** and
  a **quality prior** are for — hence the blend.
- Chose **Pearson** (mean-centred) similarity for user-user CF to correct for generous vs
  harsh raters — then flagged its own weakness (unreliable on few co-rated films) and
  decided to *test* it rather than assume it.

**Output.** `PLAN.md` (architecture) and `EXPERIMENTS.md` (every open decision + the
experiment that resolves it). The core principle written down: **start simple, add
complexity only if it improves a metric; report what doesn't.**

---

## Phase 2 — Research pass #1 (foundations)

**Where the idea came from.** Before building, we ran a deep-research pass (fan-out web
search → fetch → adversarial fact-check → synthesis) to check our approach against the
literature.

**What it returned (verified claims):**
- Hybrid-with-weighted-blend is textbook (Burke 2002's "weighted hybrid").
- Blending beats either signal alone (STAR, Meta-Prod2Vec).
- **CF usually deserves more weight than content** (Meta-Prod2Vec's best blend put ~0.15 on
  the embedding side).
- **Use a temporal split, not random/leave-one-out**, to avoid leaking future ratings
  (Ji et al., ACM TOIS).
- A known flaw: a fixed linear blend assumes each technique's value is uniform across items
  (Burke) — which motivates per-item adaptive weighting as a future change.

**Why we applied it.** It confirmed the architecture was standard (not improvised),
told us to expect CF to dominate the blend, and gave us a *cited reason* to use a temporal
split in the evaluation.

**Result.** The design was validated and sharpened; three decisions became citable.

---

## Phase 3 — Build the system (Part 1)

**What we did, file by file, each smoke-tested before moving on:**

| Step | File | Output when run |
|---|---|---|
| Data layer | `data_layer.py` | 610 users, 5,135 movies, 74,064 ratings; title lookup returns MISS for absent films (the decline path) |
| Plot embeddings | `build_embeddings.py` | 5,135 × 384 vectors cached to `embeddings.npy` (~3 min, CPU) |
| Engine 1 (content) | `content_engine.py` | Top-10 that is *on-theme but obscure* (Psycho III, Friday 13th for a Se7en fan) |
| Engine 2 (CF) | `cf_engine.py` | Top-10 of *acclaimed classics* (Die Hard, Godfather, Blade Runner); opinion feature works; declines on sparse films |
| Blend + guardrails | `recommender.py` | Hybrid top-10 (Fish Called Wanda, Godfather, Trainspotting…); all 3 behaviours pass |

**Key observation during the build (unprompted evidence):** the content engine surfaced
obscure plot-matches while CF surfaced quality classics — a live demonstration of exactly
the "plot similarity ≠ quality" weakness we'd predicted, and the live justification for
blending.

**Output.** A working engine with the three required behaviours: personalised top-10,
"what do people like me think of X?", and declining on unknown-user / film-not-in-catalogue.

---

## Phase 4 — Evaluate it (Part 2, the core)

**What we did.** Built `evaluate.py`: a temporal split (hide each user's most recent ~20%
of ratings), measured ranking (Recall/NDCG/Hit@K) against five baselines, rating prediction
(MAE + coverage, three CF variants), and guardrail precision/recall — over 150 sampled test
users, with blend weights tuned on a *disjoint* 75-user validation split.

**Output — the honest headline finding:**
- **CF works:** NDCG@10 **0.135 vs 0.063** for most-popular; a liked film in the top-10 for
  **46%** of users. A real, measured win over naive.
- **The hybrid did NOT beat CF-only** (0.133 vs 0.135), and the weight sweep **zeroed out
  the content weight**. Content-only barely beat random.
- **Why (verified, not a bug):** content ranks the held-out liked films at median position
  ~2,250/5,100 — it surfaces the long tail, which the popularity-biased offline metric
  can't reward.
- **Rating prediction:** user-user CF beat baselines by ~15% (MAE 0.685); Pearson marginally
  beat raw cosine (confirming the mean-centring decision was right to *test*).
- **Guardrails:** precision 1.0, recall 1.0.
- Added a full **"what this evaluation cannot show"** section (offline ≠ satisfaction,
  popularity bias, no diversity metric, small catalogue).

**Why this is the valuable part.** The brief rewards an honest negative result with
evidence. We built something, measured it rigorously, found our clever half didn't help,
proved *why*, and reported it straight.

---

## Phase 5 — Assistant + write-up

**What we did.** `assistant.py` — a reproducible deterministic CLI over the engine (plus an
optional Claude `claude-opus-4-8` tool-calling path), saved sample transcripts, and wrote
`REPORT.md` (Parts 2–5) + `README_run.md`.

**Output.** `transcripts/sample_session.txt` showing all three behaviours including both
decline cases; a 4-part report written for the right audiences (engineer + non-technical
client).

---

## Phase 6 — Enhancement research (recent techniques)

**Where the idea came from.** You asked whether recent papers offered innovations we could
apply. We ran a second deep-research pass, scoped to *2024–2026, training-free / CPU-only,
works with plot text + ratings*.

**What it returned (verified), and what we decided:**
- **EASE** (closed-form item-item) — flagged as a strong, cheap baseline we were
  under-shooting. → **Apply.**
- **STAR** — training-free; its retrieval stage adds **temporal decay + recency-limited
  history** over our blend. → **Apply the ingredients.**
- **LLM-reranking a CF candidate list** — *actively risky*: two 2025 papers show reranking a
  homogeneous CF top-N can *lower* accuracy; needs a paid API. → **Reject** (with reason).
- **LLM2Rec / CoLLM** — need GPU fine-tuning. → **Reject** (infeasible here).
- A verified claim that *text embeddings don't encode collaborative signal* — → used as a
  **citation** explaining our content finding.

**Why.** It gave us one concrete, free, high-value addition (EASE) and, just as valuably,
talked us *out* of the flashy options for sound reasons.

---

## Phase 7 — Enhancement #1: EASE

**What we did.** Implemented `ease.py` (one matrix inverse, fits in seconds), tuned its one
hyperparameter (λ=250) on the validation split, added it as a ranking-table row.

**Result — a genuine trade-off, not a clean win:**

| | Recall@10 | Recall@20 | NDCG@10 | Hit@10 |
|---|---|---|---|---|
| neighbourhood CF | 0.112 | 0.164 | **0.135** | 0.46 |
| **EASE** | **0.114** | **0.167** | 0.126 | **0.48** |

EASE **wins on recall and hit-rate** (surfaces a relevant film for more users) but **loses
on NDCG** (neighbourhood CF orders the very top more sharply). Neither dominates. We report
the trade-off and recommend EASE as the default (better coverage, scales far better), with
neighbourhood CF kept as an A/B candidate.

---

## Phase 8 — Enhancement #2: STAR ingredients (EXP-9)

**What we did.** Implemented `star_experiment.py` to test STAR's two additions — temporal
decay and recency-limited history — against the no-decay baseline, same split and users.

**Result — one ingredient helps a little, one doesn't transfer:**
- **Temporal decay** (λ=0.7): Recall@10 **0.119 vs 0.112** (+6%), at a −2% NDCG cost — the
  same recall↔NDCG shape as EASE. Mildly worth it.
- **Recency-limiting** (keep only last-3, STAR's best for Amazon): **hurts** here (NDCG 0.119
  vs 0.135). Movie taste is stable, unlike sequential e-commerce — so you can't copy a
  paper's hyperparameters across domains.
- Re-adding content (a=0.5) lowered everything — a **third** confirmation that content
  doesn't help top-N ranking here.

**Why this is strong.** It shows we can take a recent paper, implement it, and find that
one idea partially transfers while another doesn't — *with a domain-specific reason* — rather
than cargo-culting the paper.

---

## What we ended with

**A coherent, evidence-backed narrative:**
naive baseline → neighbourhood CF (real win) → EASE (recall/NDCG trade-off) →
recent techniques tested and *partly rejected with reasons*. Every claim is reproducible.

**The findings all reinforce each other:**
- CF-family methods work (~2× popularity); **EASE is the best default** (recall + scale).
- Content does **not** help top-N ranking — confirmed **three independent ways** (blend
  sweep zeroing it, the content ablation, and the STAR a=0.5 run) — and *why* is published.
- Recent LLM-heavy techniques were correctly ruled out for a CPU/no-training/small-data
  setting; the cheap ones (EASE, temporal decay) gave marginal, honestly-reported gains.

**Deliverables in `Test/`:**
- *Engine:* `data_layer.py`, `build_embeddings.py`, `content_engine.py`, `cf_engine.py`,
  `ease.py`, `recommender.py`
- *Assistant:* `assistant.py` (+ transcripts)
- *Evaluation:* `evaluate.py`, `star_experiment.py`, `eval_results.json`
- *Docs:* `REPORT.md` (Parts 2–5), `README_run.md`, `PLAN.md`, `EXPERIMENTS.md`, this file

**The one-line story for the interview:** *"I built a hybrid, measured it honestly, found the
clever half didn't help and proved why, then used recent research to add a stronger baseline
(EASE) and to reject the hyped options for sound reasons — and reported every result,
including the ones that went against what I set out to build."*
