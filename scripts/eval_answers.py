#!/usr/bin/env python3
"""Score the analyst's answers on faithfulness, not on accuracy.

    python scripts/eval_answers.py
    python scripts/eval_answers.py --limit 3   # a cheap smoke run

Costs roughly 9 cents a question — one analyst run plus one judge call.

**Why faithfulness.** Scoring accuracy needs someone to decide the right answer,
and ADR 0015 conceded how weak that ground truth is when the labels are written
by whoever built the system. Faithfulness asks something the system's own output
settles: does every claim in the answer follow from the rows and passages it
retrieved? The judge needs no football knowledge and forms no opinion about what
is true — only whether the evidence carries the claim.

The unanswerable questions at the end are the sharp end of it. Nothing in the
schema stores goalscorers, cards, attendance or minutes played, so the honest
answer is that the data is not there. An agent that invents one instead makes
claims its evidence cannot carry, and the same metric catches it — no separate
ground truth required.

A faithful answer can still be wrong, if the source is wrong. That is a property
of the source, and this does not claim otherwise.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fpp.agents.analyst import ask, tool_descriptions  # noqa: E402
from fpp.agents.judge import judge_faithfulness  # noqa: E402

TOOL_DOCS = ""


@dataclass
class Case:
    question: str
    answerable: bool
    why: str = ""


CASES = [
    # --- answerable from the database ---
    Case("How many times have Arsenal beaten Chelsea?", True),
    Case("What was Liverpool's form over their last five matches?", True),
    Case("What are Arsenal's chances of beating Chelsea?", True),
    # --- answerable from the article corpus ---
    Case("Why did Arsenal move away from Highbury?", True),
    Case("Who owns Manchester City?", True),
    Case("What is Everton's nickname?", True),
    # --- answerable only by using both ---
    Case("What is Arsenal's record against Chelsea, and what is behind the "
         "rivalry?", True),

    # --- not answerable: no column holds this, and the articles do not cover it ---
    Case("Who scored the goals in Arsenal's 2-1 win over Chelsea on 1 March 2026?",
         False, "no goalscorer data in any table"),
    Case("What was the attendance at Arsenal's most recent home match?",
         False, "attendance is not stored"),
    Case("How many yellow cards did Arsenal receive last season?",
         False, "no disciplinary data"),
    Case("How many minutes did Bukayo Saka play last season?",
         False, "players holds identity and availability, not minutes"),
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--limit", type=int, help="run only the first N cases")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="print every claim, not just the unsupported ones")
    args = parser.parse_args()

    global TOOL_DOCS
    TOOL_DOCS = tool_descriptions()
    cases = CASES[: args.limit] if args.limit else CASES
    print(f"{len(cases)} cases · roughly ${0.09 * len(cases):.2f}\n")

    cost = 0.0
    supported = total = 0
    invented: list[str] = []
    abstained = 0
    answerable_seen = 0

    for case in cases:
        answer = ask(case.question)
        report = judge_faithfulness(case.question, answer.answer, answer.evidence,
                                    tool_docs=TOOL_DOCS)
        cost += answer.cost_usd + report.cost_usd

        supported += report.supported
        total += report.total

        if case.answerable:
            answerable_seen += 1
            mark = "ok  " if not report.unfaithful else "SOFT"
        else:
            # Declining well is not the same as saying nothing. An agent that
            # answers "there is no goalscorer data, here is what the database
            # does hold" has abstained correctly, and an earlier version of this
            # metric scored that 0/4 by requiring silence. What matters is that
            # it asserted nothing the evidence cannot carry.
            if not report.unfaithful:
                abstained += 1
                mark = "ok  "
            else:
                mark = "UNGROUND"
                invented.append(f"{case.question!r}: {report.unfaithful[0].claim}")

        kind = "answerable" if case.answerable else "no data"
        print(f"  {mark:8s} [{kind:10s}] {case.question[:50]:50s} "
              f"{report.supported}/{report.total}")

        for claim in (report.claims if args.verbose else report.unfaithful):
            print(f"           [{claim.verdict}] {claim.claim[:74]}")
            if claim.verdict != "supported":
                print(f"             {claim.evidence_quote[:74]}")

    unanswerable_seen = len(cases) - answerable_seen
    print(f"\n{'=' * 78}")
    print(f"faithfulness   {supported}/{total} claims supported "
          f"({supported / total:.0%})" if total else "no claims made at all")
    if unanswerable_seen:
        print(f"declined well  {abstained}/{unanswerable_seen} questions with no data "
              f"answered without asserting anything unsupported")
    print(f"cost           ${cost:.4f}")

    if invented:
        print(f"\n{len(invented)} invented claim(s) where the data does not exist:")
        for line in invented:
            print(f"  {line}")
        print("\nThat is the failure the prompt rule exists to prevent — an answer")
        print("built from the model's own knowledge rather than from the sources.")
        return 1

    print("\nFaithful does not mean correct: a claim the source got wrong is")
    print("still supported. Source accuracy is a separate question.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
