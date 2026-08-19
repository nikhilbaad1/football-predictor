#!/usr/bin/env python3
"""Build the text corpus from Wikipedia.

    python scripts/rag_ingest.py               # current PL clubs + recent seasons
    python scripts/rag_ingest.py --no-cache    # re-fetch instead of reusing disk

Pages are cached under data/raw/wikipedia/, so a second run costs nothing and
does not touch the API. Wikipedia rate-limits bursts; requests are spaced.

Only text with no schema goes in. Scorelines and dates live in `matches` and are
queried exactly — see ADR 0003 on why retrieval is the wrong tool for those.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlalchemy import text  # noqa: E402

from fpp.db import get_engine, init_db  # noqa: E402
from fpp.rag.wikipedia import fetch_page, resolve_club_page, store_document  # noqa: E402

# Season articles carry the narrative a club article does not: who led, what
# turned, who went down. No team metadata — they are about all of them.
SEASON_PAGES = {
    "2023–24 Premier League": "2324",
    "2024–25 Premier League": "2425",
    "2025–26 Premier League": "2526",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--no-cache", action="store_true", help="re-fetch every page")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    engine = get_engine()
    init_db(engine)
    session = requests.Session()

    with engine.connect() as conn:
        clubs = [r[0] for r in conn.execute(
            text("""SELECT DISTINCT t.name FROM teams t
                    JOIN players p ON p.team_id = t.id ORDER BY t.name""")
        )]
    if not clubs:
        print("No clubs with players. Run scripts/fpl.py first.")
        return 1

    print(f"{len(clubs)} clubs · {len(SEASON_PAGES)} season articles\n")

    chunks = docs = 0
    unresolved: list[str] = []

    for club in clubs:
        page = resolve_club_page(club, session) if not args.no_cache else None
        if page is None and args.no_cache:
            page = resolve_club_page(club, session)
        if page is None:
            unresolved.append(club)
            continue
        n = store_document(page, engine, team=club)
        chunks += n
        docs += 1
        print(f"  {club:26s} -> {page['title']:34s} {n:3d} chunks")

    for title, season in SEASON_PAGES.items():
        page = fetch_page(title, session)
        if page is None:
            print(f"  (missing season article: {title})")
            continue
        n = store_document(page, engine, season=season)
        chunks += n
        docs += 1
        print(f"  {'[season]':26s} -> {page['title']:34s} {n:3d} chunks")

    print(f"\n{docs} documents · {chunks} chunks")

    if unresolved:
        print(f"\n  WARNING: no article found for {len(unresolved)} club(s):")
        for club in unresolved:
            print(f"    {club!r}")
        print("  Not guessed at — a plausible wrong article retrieves well and")
        print("  answers nothing. Add the title explicitly if you know it.")

    print("\nText from Wikipedia, CC BY-SA 4.0.")
    print("Next: python scripts/rag_query.py \"who won the title\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
