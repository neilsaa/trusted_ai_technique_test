"""Layer B — the conversational assistant over the hybrid recommender.

Two ways to run it:

  * Deterministic CLI (default, no API key needed) — a small intent router maps a user's
    natural-language question to one of the engine functions and narrates the result.
    This is what produces the reproducible transcripts in transcripts/.

      python assistant.py --user 1 --ask "what should I watch tonight?"
      python assistant.py --demo                     # runs the scripted scenarios

  * Claude tool-calling agent (optional) — if the `anthropic` SDK and an API key are
    available, `--llm` routes through Claude with the engine functions exposed as tools.
    Claude only decides which tool to call and how to phrase the reply; it cannot invent
    films (the tools are the sole source of truth), and it must decline when a tool
    returns nothing. Model: claude-opus-4-8 with adaptive thinking.

The engine (recommender.py) is identical in both paths — only the routing differs.
"""

from __future__ import annotations

import argparse
import re

from recommender import HybridRecommender, build_recommender

MODEL = "claude-opus-4-8"

SYSTEM_PROMPT = """You are a film-discovery assistant for a movie catalogue service.
A user identifies themselves by a numeric user id; use their history to personalise.

Rules you must follow:
- Only recommend or discuss films returned by the tools. Never invent a title, year, or
  rating. If a tool returns no data, say so plainly and do not guess.
- If the user or film is not found, decline and explain why — do not fabricate an answer.
- When you give recommendations, briefly say why (e.g. the user's taste or similar users).
- Be concise and concrete. One short paragraph plus a list is plenty.
"""


# ---------------------------------------------------------------------------
# Deterministic intent router (no LLM) — the reproducible path
# ---------------------------------------------------------------------------
class DeterministicAssistant:
    def __init__(self, rec: HybridRecommender):
        self.rec = rec

    def answer(self, user_id: int, text: str) -> str:
        t = text.lower().strip()

        # intent: opinion about a specific film ("what do people like me think of X")
        m = (re.search(r"think of (.+)", t) or re.search(r"opinion[s]? (?:on|of) (.+)", t)
             or re.search(r"should i watch ([^?]+)\?*$", t) if "tonight" not in t else None)
        if ("think of" in t or "opinion" in t
                or ("what do people" in t and " of " in t)):
            title = self._extract_title(text)
            if title:
                return self._opinion(user_id, title)

        # intent: tonight / recommendation
        if any(k in t for k in ["watch tonight", "what should i watch", "recommend",
                                 "suggest", "something to watch", "what to watch"]):
            return self._recommend(user_id)

        # intent: profile
        if any(k in t for k in ["my taste", "my profile", "what do i like", "my history"]):
            return self._profile(user_id)

        # fallback: treat as a recommendation request
        return self._recommend(user_id)

    def _extract_title(self, text: str) -> str | None:
        for pat in [r"think of (.+?)\??$", r"opinion[s]? (?:on|of) (.+?)\??$",
                    r"of ([A-Z][^?]+)\??$"]:
            m = re.search(pat, text.strip(), re.IGNORECASE)
            if m:
                return m.group(1).strip().rstrip("?").strip()
        return None

    def _recommend(self, user_id: int) -> str:
        res = self.rec.recommend_for_user(user_id, 10)
        if not res.ok:
            return f"I can't help with that: {res.reason}."
        lines = []
        if res.mode.startswith("fallback"):
            lines.append(f"Note: {res.reason}.")
        prof = self.rec.get_user_profile(user_id)
        if prof and res.mode == "personalised":
            liked = ", ".join(t for t, _ in prof["top_rated"][:3])
            lines.append(f"Based on your taste (you rated {liked} highly), here are 10 picks:")
        else:
            lines.append("Here are 10 popular, well-rated films:")
        for i, it in enumerate(res.items, 1):
            lines.append(f"  {i:>2}. {it.title} ({it.year}) — {it.reason}")
        return "\n".join(lines)

    def _opinion(self, user_id: int, title: str) -> str:
        r = self.rec.similar_users_opinion(user_id, title)
        if not r["ok"]:
            return f"I can't answer that: {r['reason']}."
        return (f"People with taste like yours rate **{r['title']} ({r['year']})** about "
                f"**{r['predicted_rating']:.1f}/5** — {r['reason']}.")

    def _profile(self, user_id: int) -> str:
        p = self.rec.get_user_profile(user_id)
        if not p:
            return f"I have no history for user {user_id}."
        top = ", ".join(f"{t} ({r})" for t, r in p["top_rated"][:5])
        return (f"User {user_id}: {p['n_ratings']} ratings, average {p['mean_rating']}. "
                f"Favourite genres: {', '.join(p['top_genres'])}. Top-rated: {top}.")


# ---------------------------------------------------------------------------
# Optional Claude tool-calling agent
# ---------------------------------------------------------------------------
def run_llm(rec: HybridRecommender, user_id: int, text: str) -> str:
    import anthropic
    from anthropic import beta_tool

    @beta_tool
    def recommend_for_user(k: int = 10) -> str:
        """Return the top-k personalised film recommendations for the current user.

        Args:
            k: how many films to return (default 10).
        """
        res = rec.recommend_for_user(user_id, k)
        if not res.ok:
            return f"DECLINE: {res.reason}"
        head = f"mode={res.mode}" + (f" ({res.reason})" if res.reason else "")
        rows = [f"{it.title} ({it.year}) score={it.score:.3f} :: {it.reason}" for it in res.items]
        return head + "\n" + "\n".join(rows)

    @beta_tool
    def similar_users_opinion(title: str) -> str:
        """Predict how users with taste similar to the current user rate a given film.

        Args:
            title: the film title to look up.
        """
        r = rec.similar_users_opinion(user_id, title)
        if not r["ok"]:
            return f"DECLINE: {r['reason']}"
        return (f"{r['title']} ({r['year']}): predicted {r['predicted_rating']:.2f}/5 "
                f"from {r['n_support']} taste-neighbours (their avg {r['neighbour_mean']:.2f})")

    @beta_tool
    def get_user_profile() -> str:
        """Return the current user's rating count, mean rating, top genres and top films."""
        p = rec.get_user_profile(user_id)
        if not p:
            return f"DECLINE: user {user_id} has no history"
        return (f"{p['n_ratings']} ratings, mean {p['mean_rating']}, "
                f"genres {p['top_genres']}, top {p['top_rated']}")

    client = anthropic.Anthropic()
    runner = client.beta.messages.tool_runner(
        model=MODEL,
        max_tokens=1024,
        thinking={"type": "adaptive"},
        system=SYSTEM_PROMPT + f"\nThe current user id is {user_id}.",
        tools=[recommend_for_user, similar_users_opinion, get_user_profile],
        messages=[{"role": "user", "content": text}],
    )
    final = None
    for message in runner:
        final = message
    if final is None:
        return "(no response)"
    return "".join(b.text for b in final.content if b.type == "text")


# ---------------------------------------------------------------------------
DEMO_SCRIPT = [
    (1, "What should I watch tonight?"),
    (1, "What do people with taste like mine think of Pulp Fiction?"),
    (1, "What do people like me think of The Matrix?"),        # film not in catalogue
    (999999, "What should I watch tonight?"),                  # unknown user
    (414, "Recommend me something to watch"),
    (1, "What's my taste profile?"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", type=int)
    ap.add_argument("--ask", type=str)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--llm", action="store_true", help="route through Claude (needs API key)")
    args = ap.parse_args()

    rec = build_recommender()
    det = DeterministicAssistant(rec)

    def respond(uid, text):
        if args.llm:
            return run_llm(rec, uid, text)
        return det.answer(uid, text)

    if args.demo:
        for uid, text in DEMO_SCRIPT:
            print(f"\n[user {uid}] {text}")
            print(respond(uid, text))
        return

    if args.user is None or not args.ask:
        print("usage: --user ID --ask 'question'   (or --demo)")
        return
    print(respond(args.user, args.ask))


if __name__ == "__main__":
    main()
