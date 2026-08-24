#!/usr/bin/env python3
"""Score the analyst agent on routing: did it use the right source?

    python scripts/eval_analyst.py
    python scripts/eval_analyst.py --limit 4   # a cheap smoke run

Costs real money — roughly 5 cents a question, so about 60 cents for the set.

**What makes this ground truth better than the retrieval eval's.** ADR 0015
admitted that the retrieval cases were authored by the same person who built the
retriever, after reading the corpus, which makes those relevance labels
contestable. These labels are not judgement calls: a question is `structured`
when the fact it asks for has a column in the schema, and `text` when it does
not. "How many times has X beaten Y" is a COUNT over `matches`. "What is the
club's nickname" has no column anywhere and can only come from prose. The label
is checkable against `schema.sql` rather than against an opinion.

What this does NOT measure is whether the answer is correct — only whether the
agent went to the source that could answer it. Answer quality needs a separate
harness (PLAN section 7 wants Ragas for that) and is not claimed here.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fpp.agents.analyst import ask  # noqa: E402


@dataclass
class Case:
    question: str
    route: str            # structured | text | both
    why: str              # what makes that the right route


CASES = [
    # --- structured: the fact has a column ---
    Case("How many times have Arsenal beaten Chelsea?", "structured",
         "a COUNT over matches.result"),
    Case("What are Arsenal's chances of beating Chelsea?", "structured",
         "the model's output, in predictions"),
    Case("What was Liverpool's form over their last five matches?", "structured",
         "matches.result ordered by date"),
    Case("Which Premier League fixtures are scheduled next?", "structured",
         "unplayed rows in matches"),
    Case("Which Arsenal players are currently unavailable?", "structured",
         "player_availability.status"),
    Case("How many matches are in the database in total?", "structured",
         "COUNT(*) over matches"),

    # --- text: no column holds this ---
    Case("Why did Arsenal move away from Highbury?", "text",
         "reasons are prose; no column stores them"),
    Case("What is Everton's nickname and where does it come from?", "text",
         "no nickname column exists"),
    Case("Who owns Manchester City?", "text",
         "ownership is not in the schema"),
    Case("When was Hull City founded?", "text",
         "founding dates are not in the schema"),
    Case("What is the rivalry between Liverpool and Everton called?", "text",
         "rivalry names are prose"),

    # --- both: one half is countable, the other is not ---
    Case("What is Arsenal's record against Chelsea, and what is the history "
         "behind the rivalry?", "both",
         "the record is a count; the history is prose"),
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--limit", type=int, help="run only the first N cases")
    args = parser.parse_args()

    cases = CASES[: args.limit] if args.limit else CASES
    print(f"{len(cases)} cases · roughly ${0.05 * len(cases):.2f}\n")

    correct = 0
    cost = 0.0
    misroutes: list[str] = []

    for case in cases:
        result = ask(case.question)
        cost += result.cost_usd
        ok = result.route == case.route

        # A question needing both is still served if the agent used both. A
        # question needing one is NOT wrong for also consulting the other —
        # but it is wrong for consulting only the other one.
        acceptable = ok or (case.route in {"structured", "text"} and result.route == "both")

        if acceptable:
            correct += 1
            mark = "ok  " if ok else "ok+ "
        else:
            mark = "WRONG"
            misroutes.append(f"{case.question!r}: wanted {case.route}, got {result.route}")

        tools = ", ".join(dict.fromkeys(result.tools_used)) or "none"
        print(f"  {mark:5s} {case.question[:56]:56s} {result.route:10s} [{tools}]")
        if not acceptable:
            print(f"        should be {case.route}: {case.why}")

    n = len(cases)
    print(f"\n{'=' * 78}")
    print(f"routed correctly  {correct}/{n}")
    print(f"cost              ${cost:.4f}")
    print("\n'ok+' means it also consulted the other source, which is not an error.")

    if misroutes:
        print(f"\n{len(misroutes)} misroute(s):")
        for m in misroutes:
            print(f"  {m}")
        print("\nA question sent to the wrong source gets a fluent answer built on")
        print("whatever that source had — which is the failure ADR 0003 exists to stop.")
        return 1

    print("\nEvery question reached a source that could answer it. This does not")
    print("measure whether the answers are right — that needs its own harness.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
