#!/usr/bin/env python3
"""Download and load match data.

    python scripts/ingest.py                      # top-5 leagues, default seasons
    python scripts/ingest.py --divisions E0       # Premier League only
    python scripts/ingest.py --seasons 2425 2526  # specific seasons
    python scripts/ingest.py --no-cache           # re-download

Files are cached under data/raw/ — fetch once, reuse forever.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlalchemy import text  # noqa: E402

from fpp.config import DEFAULT_SEASONS, DIVISIONS  # noqa: E402
from fpp.db import get_engine, init_db  # noqa: E402
from fpp.ingest import ingest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--divisions", nargs="+", default=list(DIVISIONS),
                        help=f"any of {', '.join(DIVISIONS)}")
    parser.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    engine = get_engine()
    init_db(engine)

    counts = ingest(args.divisions, args.seasons, engine, use_cache=not args.no_cache)
    if not counts:
        print("No data ingested. Check the network and that the season codes exist.")
        return 1

    with engine.connect() as conn:
        matches = conn.execute(text("SELECT COUNT(*) FROM matches")).scalar_one()
        teams = conn.execute(text("SELECT COUNT(*) FROM teams")).scalar_one()
        with_odds = conn.execute(text("SELECT COUNT(*) FROM odds")).scalar_one()
        span = conn.execute(
            text("SELECT MIN(match_date), MAX(match_date) FROM matches")
        ).fetchone()

    print(f"\n{matches:,} matches · {teams} teams · {with_odds:,} with closing odds")
    print(f"span: {span[0]} to {span[1]}")
    print("\nNext: python scripts/backtest.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
