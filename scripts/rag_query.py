#!/usr/bin/env python3
"""Search the text corpus.

    python scripts/rag_query.py "who won the title in 2024-25"
    python scripts/rag_query.py "stadium capacity" --team Arsenal --mode hybrid
    python scripts/rag_query.py "relegated" --season 2425 -k 3

Three modes. `lexical` is BM25 and free. `dense` is Voyage embeddings and costs
one query embedding. `hybrid` fuses both with Reciprocal Rank Fusion and is the
default, because a passage both rankers surface is better evidence than one
either ranker alone is enthusiastic about.

Metadata filters run before ranking, which is the first stage of the cascade in
PLAN section 4. Scoping to the wrong club returns fluent, well-ranked, wrong text.

Which mode actually wins is a measured question, not an assumed one — see
scripts/eval_retrieval.py and ADR 0015.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fpp.rag import search  # noqa: E402
from fpp.rag.dense import dense_search  # noqa: E402
from fpp.rag.hybrid import hybrid_search  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("query", help="what to search for")
    parser.add_argument("--team", help="restrict to one club's documents")
    parser.add_argument("--season", help="restrict to one season, e.g. 2425")
    parser.add_argument("-k", "--limit", type=int, default=5)
    parser.add_argument("--mode", choices=["lexical", "dense", "hybrid"], default="hybrid")
    parser.add_argument("--full", action="store_true", help="print whole passages")
    args = parser.parse_args()

    common = dict(team=args.team, season=args.season, limit=args.limit)
    runner = {"lexical": search, "dense": dense_search, "hybrid": hybrid_search}[args.mode]
    hits = runner(args.query, **common)
    if not hits:
        print("No passage matched. The corpus is Wikipedia club and season articles —")
        print("scorelines and dates are not in here; those are queried from `matches`.")
        return 0

    scope = " · ".join(filter(None, [args.team, args.season])) or "whole corpus"
    print(f'"{args.query}"  ({args.mode} · {scope})\n')

    for hit in hits:
        heading = f" › {hit['heading']}" if hit["heading"] else ""
        if "rrf_score" in hit:
            # Say how many rankers found it. A fused score nobody can account
            # for is less useful than two separate rankings.
            marker = f"{hit['rrf_score']:.4f} ({hit['found_by']}/2)"
        else:
            marker = f"{hit['score']:6.2f}"
        print(f"  [{hit['rank']}] {marker}  {hit['title']}{heading}")
        body = hit["text"] if args.full else hit["text"][:300] + "…"
        for line in textwrap.wrap(body, width=88):
            print(f"        {line}")
        print(f"        {hit['url']}")
        print()

    print("Text from Wikipedia, CC BY-SA 4.0.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
