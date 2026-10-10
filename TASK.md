# TrustedAI — Middle AI Engineer Test

## About this exercise

We are not testing how much you can build in an afternoon. We are testing how you
think about a system: whether you can tell if it works, decide whether it is ready,
and hand it to someone who has to trust it.

The build is deliberately small. Most of what we score is your reasoning about it.

| Section | Weight | Rough time |
|---|---|---|
| Part 2 — Evaluation | 35 | ~75 min |
| Part 3 — Engineering note | 25 | ~30 min |
| Part 4 — Client handover note | 20 | ~20 min |
| Part 1 — System | 15 | ~60 min |
| Part 5 — Note on AI use | 5 | ~5 min |

**About 4 hours.** That is our honest estimate, not a floor to beat. We would rather
read a focused 4-hour submission than a polished 10-hour one, and we will ask how
long it took — answer honestly, it does not count against you.

**If you run short**, reduce Part 1 first and say what you reduced. A retrieval-only
system with no agent loop is an acceptable Part 1. Cutting a clearly-labelled corner
costs you very little; a thin Part 2 costs you a lot.

**On AI tools:** use them heavily. That is the job. We do not penalise AI use, we
reward *managed* AI use — knowing what to ask for, checking what comes back, and
owning the result. In the interview you will be asked to explain your decisions.
"The AI wrote it" is not an answer.

**A few things worth saying plainly**, so you do not waste time guessing:

- Telling us what your evaluation *cannot* show scores **positively**. Hedging that
  is specific to your own results is a strength here, not a weakness.
- If your system turns out to be no better than a naive approach, reporting that
  clearly is a **good** outcome for you. We are not scoring the win.
- This is a scored exercise on a public dataset. Nothing you submit is used in any
  commercial work.

---

## The situation

A client runs a film catalogue service. They want to add an assistant that helps
their users find something to watch, and they want it to feel like it knows them
rather than like a search box.

You are writing the PoC we would use to explore this with them. If it goes well,
some version eventually ships to their users, and someone has to sign an agreement
about how well it works.

Keep that client in mind. Several questions below only have good answers if you have
thought about who is on the other end.

## Dataset

In `data/ml-latest-small-filtered/`:

| File | Description |
|------|-------------|
| `movies_with_plots.csv` | 5,135 movies — `movieId`, `title`, `year`, `genres`, `plot` (avg ~3,200 chars) |
| `ratings.csv` | 74,064 ratings from 610 users — `userId`, `movieId`, `rating` (0.5–5.0), `timestamp` |
| `tags.csv` | 2,440 user-generated tags — sparse |
| `links.csv` | IMDb / TMDB ids |
| `movies.csv` | Basic movie info without plots |

Notes on the data:

- Movies span 1903–2014. This is a filtered subset of MovieLens; some well-known
  films are absent because the source had no plot for them.
- ~51% of movies have fewer than 5 ratings. The sparsity is on the movie side —
  users average ~121 ratings each.
- Tags are sparse. Do not build your approach around tags alone.

Run `python scripts/verify_dataset.py` to check your copy is intact.

**Cost:** if you use a paid API, a few dollars is plenty for this exercise — you do
not need to run anything expensive across the whole dataset. If cost is a barrier at
all, tell us and we will provide a key. Do not spend your own money to do well here.

---

## Part 1 — Build it  (~60 min, 15 pts)

An assistant that helps a user discover films. A user identifies themselves by user
id, and the assistant uses their history to personalise.

It should be able to:

1. Answer a question that depends on who is asking
   — *"what should I watch tonight?"*
2. Answer a question that needs more than one lookup
   — *"what do people with taste like mine think of Pulp Fiction?"*
3. Decline when the data cannot support an answer — for example a film outside the
   catalogue, or a user id with no history

That is the whole functional bar. **No UI, no API, no packaging, no Docker.** A
script or a notebook is fine. We are not scoring how elaborate your pipeline is —
Part 1 is worth the fewest points of any section on purpose.

How you build it is up to you.

## Part 2 — Evaluate it  (~75 min, 35 pts)

Show us how well it works, with evidence.

Your evaluation must be **reproducible and grounded in the data** — not only in
hand-picked examples. We will run your evaluation code ourselves, so include it and
tell us how to run it. Report how many users or items each number is computed over.

What you measure, how you obtain something to measure against, and what you conclude
are all your decisions to make and defend. Tell us what your evaluation *cannot*
tell you as clearly as what it can.

Sampling is fine — you do not need to run your full pipeline over every user. Say
what you sampled and why.

## Part 3 — Engineering note  (~30 min, 25 pts, writing only)

No implementation. Using your own results:

1. **Would you ship this to the client's users?** Why, or why not.
2. **The next change you would make.** What you predict it improves, what it would
   cost or break, and what evidence would convince you it actually worked.
3. **What changes between this and production.** What you would monitor and at what
   threshold, what breaks first when this serves 100× more users, and what would
   make you roll it back.

Be specific. "Add caching and monitoring" is not an answer; what you would cache,
and what you would alert on at what number, is. We give you no pricing or traffic
figures — if you reason about cost, state your assumptions and we will judge the
reasoning, not the numbers.

## Part 4 — Client handover note  (~20 min, 20 pts, writing only)

One page, written for the client — a business owner, not an engineer. No jargon.

- What accuracy or quality do we commit to, and how is it measured?
- What does the system explicitly *not* do?
- How can the client check the claim themselves, without taking our word for it?
- What happens when it gets something wrong?

This is the document that would go into a real handover. Write it like it counts.

## Part 5 — Note on AI use  (~5 min, 5 pts)

A short paragraph: which parts did you use an AI assistant for, what did it get
right, what did it get wrong, and what did you override? Specific answers score
well; "it helped with boilerplate" does not.

---

## Deliverables

1. **Code** — your system and your evaluation, with instructions to run both.
2. **Report** — Parts 2, 3, 4 and 5. Markdown is fine. Roughly 2–4 pages; we are not
   counting, but length is not a proxy for depth.
3. **Sample outputs** — a transcript or saved results showing the system working, so
   we can read your results before running anything. If your solution needs an API
   key, say so here.

## Interview  (45 min)

A short demo, a walk through your report, and questions about your decisions. We
will push hardest on the evaluation.

For the demo: run it on your own machine and share your screen. You do not need to
package anything for us.

## What we care about

- Whether you know what "it works" means, and can prove it
- Honest assessment of your own system, especially where it fails
- Judgement about what matters and what does not
- Writing that a reader can act on

## What we do not care about

- State-of-the-art quality
- Infrastructure, packaging, or UI
- Exhaustive hyperparameter tuning
- How much code you wrote
