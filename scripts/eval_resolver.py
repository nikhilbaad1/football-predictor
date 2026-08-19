#!/usr/bin/env python3
"""Score the entity-resolution agent against the deterministic baseline.

    python scripts/eval_resolver.py            # run the whole set
    python scripts/eval_resolver.py --limit 3  # a cheap smoke run

Costs real money — every case is one Claude call. Roughly $0.01 per case.

The set is built from cases the deterministic resolver *refuses*, because that
is the agent's entire job; cases a rule already settles are handled without a
model call and prove nothing. Every expected answer below is a fact about real
football clubs, checked against the 34 English clubs this database knows.

The measure that matters is not accuracy. It is **wrong matches**: an agent that
says "new club" too often creates a duplicate somebody notices, while one that
matches two different clubs fuses their histories silently. The baseline scores
zero wrong matches by refusing everything here, so any wrong match is a
regression no accuracy number redeems (ADR 0012).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlalchemy import text  # noqa: E402

from fpp.agents import resolve_with_agent  # noqa: E402
from fpp.db import get_engine  # noqa: E402
from fpp.ingest.resolve import resolve_team_name  # noqa: E402

# (raw name, expected decision, expected target or None)
# Every one of these is refused by the deterministic resolver.
EVAL_SET: list[tuple[str, str, str | None]] = [
    # Spelling variants of clubs the database already knows.
    ("Brighton and Hove Albion", "matched", "Brighton & Hove Albion"),
    ("Man U", "matched", "Manchester United"),
    ("Spurs FC", "matched", "Tottenham"),
    ("Forest", "matched", "Nottingham Forest"),
    ("West Brom Albion", "matched", "West Bromwich Albion"),
    # Real clubs with no history here. These must not be matched to anything.
    ("Coventry City", "new_entity", None),
    ("Blackburn Rovers", "new_entity", None),
    ("Plymouth Argyle", "new_entity", None),
    ("Wrexham", "new_entity", None),
    # The trap. This database has Sheffield United; Sheffield Wednesday is a
    # different club in the same city. Matching them is the exact failure the
    # whole design exists to prevent.
    ("Sheffield Wednesday", "new_entity", None),
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--limit", type=int, help="run only the first N cases")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    with get_engine().connect() as conn:
        known = {r[0] for r in conn.execute(text("SELECT name FROM teams"))}

    cases = EVAL_SET[: args.limit] if args.limit else EVAL_SET
    print(f"{len(cases)} cases · {len(known)} known clubs · model calls cost real money\n")

    correct = wrong_match = escalated = 0
    cost = 0.0
    rows = []

    for raw, want_decision, want_target in cases:
        baseline = resolve_team_name(raw, known)
        assert not baseline.is_resolved, f"{raw!r} is not a baseline failure"

        got = resolve_with_agent(raw, known)
        cost += got.cost_usd

        ok = got.decision == want_decision and got.resolved_to == want_target
        # The failure that actually matters: a confident link to a *different* club.
        is_wrong_match = got.decision == "matched" and got.resolved_to != want_target

        if ok:
            correct += 1
            mark = "ok  "
        elif is_wrong_match:
            wrong_match += 1
            mark = "WRONG"
        else:
            escalated += 1
            mark = "miss"

        rows.append((mark, raw, want_decision, want_target, got))
        target = f" -> {got.resolved_to}" if got.resolved_to else ""
        print(f"  {mark:5s} {raw:26s} {got.decision}{target}")
        if not ok:
            print(f"        expected {want_decision}"
                  f"{' -> ' + want_target if want_target else ''}")
            print(f"        said: {got.reasoning}")
        if got.validation_note:
            print(f"        validation: {got.validation_note}")

    n = len(cases)
    print(f"\n{'=' * 62}")
    print(f"correct        {correct}/{n}")
    print(f"WRONG MATCHES  {wrong_match}      <- the number that decides this")
    print(f"missed         {escalated}      (new/uncertain where a match existed, or vice versa)")
    print(f"cost           ${cost:.4f}")

    print(f"\nbaseline: refuses all {n}, so 0 correct and 0 wrong matches.")
    if wrong_match:
        print("\nVERDICT: do not ship. A wrong match corrupts data silently, and the")
        print("baseline never makes one — this is strictly worse where it counts.")
        return 1
    if correct > 0:
        print(f"\nVERDICT: ship. {correct} names resolved that no rule could, still zero")
        print("wrong matches. Strictly better than refusing everything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
