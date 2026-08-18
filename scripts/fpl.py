#!/usr/bin/env python3
"""Load Premier League players and availability from the FPL API.

    python scripts/fpl.py             # ingest players + today's availability
    python scripts/fpl.py --pending   # list names no rule would commit to

Intended to run on a schedule. Each run writes an availability snapshot stamped
with today's date rather than overwriting the last one, so a backtest can ask
what was known before a match instead of what turned out to be true.

Team names arrive in a third vocabulary — FPL says "Spurs", the results files
say "Tottenham". Anything the deterministic resolver will not commit to is
reported here and its squad is skipped, because a wrong match silently merges
two clubs' players. See ADR 0012.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlalchemy import text  # noqa: E402

from fpp.db import get_engine, init_db  # noqa: E402
from fpp.ingest import ingest_fpl, pending_resolutions  # noqa: E402


def _print_pending(rows: list[dict]) -> None:
    if not rows:
        print("\nNothing pending — every source name resolved.")
        return
    print(f"\n{len(rows)} name(s) awaiting a decision:\n")
    for r in rows:
        print(f"  [{r['source']}/{r['entity_type']}] {r['raw_name']!r}")
        print(f"      {r['rationale']}")
    print(
        "\nEach is either a club with no history here, or an unrecognised spelling\n"
        "of one that does. Telling those apart needs knowledge of which clubs\n"
        "exist, which no string rule has — that is the entity-resolution agent's job."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pending", action="store_true",
                        help="list unresolved names and exit")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    engine = get_engine()
    init_db(engine)

    if args.pending:
        _print_pending(pending_resolutions(engine))
        return 0

    counts = ingest_fpl(engine)
    if not counts["players"] and not counts["unresolved"]:
        print("Nothing ingested. Check the network and that the FPL API is up.")
        return 1

    print(f"{counts['players']:,} players · {counts['availability']:,} availability rows")

    with engine.connect() as conn:
        flagged = conn.execute(
            text(
                """
                SELECT COUNT(*) FROM player_availability
                WHERE as_of = (SELECT MAX(as_of) FROM player_availability)
                  AND (status IS NOT NULL AND status <> 'a')
                """
            )
        ).scalar_one()
    print(f"{flagged:,} player(s) not fully available in today's snapshot")

    if counts["unresolved"]:
        print(f"\n  WARNING: {len(counts['unresolved'])} team(s) unresolved — "
              f"their squads were skipped:")
        for name in counts["unresolved"]:
            print(f"    {name!r}")
        print("  Run with --pending for the reasoning.")

    print("\nNext: python scripts/fpl.py --pending")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
