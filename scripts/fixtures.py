#!/usr/bin/env python3
"""Refresh upcoming fixtures and predict them.

    python scripts/fixtures.py                 # fetch fixtures, predict, store
    python scripts/fixtures.py --division SP1  # a different league
    python scripts/fixtures.py --no-predict    # just refresh the fixture list
    python scripts/fixtures.py --prune         # drop unplayed past fixtures

Intended to run on a schedule. The feed covers roughly the next week, so a
weekly run keeps the page current; running it more often is harmless because
both the ingest and the prediction write are idempotent.

Predictions are only stored for matches that have not kicked off yet. That is
what makes them evidence rather than hindsight — see ADR 0010.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fpp.config import DISPLAY_DIVISION, DIVISIONS  # noqa: E402
from fpp.db import get_engine, init_db, store_predictions  # noqa: E402
from fpp.ingest import (  # noqa: E402
    ingest_fixtures,
    prune_stale_fixtures,
    stale_fixtures,
    teams_without_history,
)
from fpp.predict import predict_upcoming  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--division", default=DISPLAY_DIVISION,
                        help=f"division to predict; any of {', '.join(DIVISIONS)}")
    parser.add_argument("--no-predict", action="store_true",
                        help="refresh the fixture list without predicting")
    parser.add_argument("--prune", action="store_true",
                        help="delete unplayed fixtures whose date has passed")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    engine = get_engine()
    init_db(engine)

    counts = ingest_fixtures(list(DIVISIONS), engine)
    skipped = counts.pop("skipped_past", 0)
    if counts:
        summary = " · ".join(f"{d} {n}" for d, n in sorted(counts.items()))
        print(f"new fixtures: {summary}")
    else:
        print("no new fixtures in the feed")
    if skipped:
        print(f"  ignored {skipped} fixture(s) already in the past")

    if args.prune:
        removed = prune_stale_fixtures(engine)
        print(f"pruned {removed} unplayed fixture(s) whose date had passed")
    else:
        backlog = stale_fixtures(engine)
        if backlog:
            print(f"  {backlog} unplayed fixture(s) sit in the past "
                  f"(postponements) — clear with --prune")

    unknown = teams_without_history(engine)
    if unknown:
        print("\n  WARNING: no match history for " + ", ".join(repr(t) for t in unknown))
        print("  Either a newly promoted club, or a name spelled differently in the")
        print("  fixtures feed than in the results files. The second is a bug: check")
        print("  ALIASES in src/fpp/ingest/teams.py for a near-duplicate before")
        print("  trusting any prediction involving these teams.")

    if args.no_predict:
        return 0

    predictions = predict_upcoming(args.division)
    if predictions.empty:
        print(f"\nNo scheduled {DIVISIONS[args.division]} fixtures to predict.")
        print("The feed covers about a week ahead, so this is expected between")
        print("rounds and before a season starts.")
        return 0

    stored = store_predictions(predictions, engine)
    print(f"\n{DIVISIONS[args.division]} — {len(predictions)} fixture(s), {stored} stored\n")
    for row in predictions.itertuples(index=False):
        print(
            f"  {row.match_date}  {row.home_team:<24} v {row.away_team:<24}"
            f"  {row.prob_home:5.1%} / {row.prob_draw:5.1%} / {row.prob_away:5.1%}"
        )
    print("\nProbabilistic estimates, not betting advice.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
