#!/usr/bin/env python3
"""Predict a single fixture.

    python scripts/predict.py "Arsenal" "Chelsea"
    python scripts/predict.py "Man City" "Liverpool" --scores
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from fpp.config import DIVISIONS  # noqa: E402
from fpp.db import load_matches  # noqa: E402
from fpp.predict import Fixture, fit_models, predict_fixtures  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("home")
    parser.add_argument("away")
    parser.add_argument("--scores", action="store_true", help="show the scoreline grid")
    args = parser.parse_args()

    matches = load_matches(divisions=list(DIVISIONS), played_only=True)
    if matches.empty:
        print("No matches found. Run scripts/ingest.py first.")
        return 1

    known = set(matches["home_team"]) | set(matches["away_team"])
    for team in (args.home, args.away):
        if team not in known:
            close = sorted(t for t in known if team.lower()[:4] in t.lower())
            print(f"Unknown team: {team!r}")
            if close:
                print("Did you mean:", ", ".join(close[:5]))
            return 1

    elo, dc = fit_models(matches)
    row = predict_fixtures([Fixture(args.home, args.away)], elo, dc).iloc[0]

    print(f"\n{args.home} v {args.away}")
    print(f"  home win  {row.prob_home:6.1%}")
    print(f"  draw      {row.prob_draw:6.1%}")
    print(f"  away win  {row.prob_away:6.1%}")
    print(f"\n  expected goals  {row.expected_goals_home} - {row.expected_goals_away}")
    print(f"  likeliest score {row.likeliest_score} ({row.likeliest_score_prob:.1%})")
    print(f"\n  elo  {elo.rating(args.home):.0f} v {elo.rating(args.away):.0f}")

    if args.scores:
        m = dc.scoreline_matrix(args.home, args.away)[:6, :6]
        print("\n  scoreline probabilities (%)")
        print("      " + "".join(f"{j:>7}" for j in range(6)))
        for i in range(6):
            print(f"    {i} " + "".join(f"{100 * m[i, j]:7.2f}" for j in range(6)))

    print("\n  Probabilistic estimate, not betting advice.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
