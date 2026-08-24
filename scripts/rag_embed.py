#!/usr/bin/env python3
"""Embed the text corpus for dense retrieval.

    python scripts/rag_embed.py            # embed anything not yet embedded
    python scripts/rag_embed.py --limit 50 # a cheap partial run
    python scripts/rag_embed.py --status   # what is embedded, no API calls

Safe to re-run: chunks that already have a vector are skipped, so this costs
nothing when there is nothing new.

**This is slow on Voyage's unbilled free tier** — 3 requests and 10,000 tokens
per minute, so ~1,000 chunks takes around a quarter of an hour. Adding a payment
method in the Voyage dashboard raises the limits and still leaves the 200M free
tokens intact, after which `BATCH` can go up and `REQUEST_DELAY` down.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlalchemy import text  # noqa: E402

from fpp.db import get_engine, init_db  # noqa: E402
from fpp.rag.dense import BATCH, MODEL, REQUEST_DELAY, embed_corpus  # noqa: E402


def _status(engine) -> tuple[int, int]:
    with engine.connect() as conn:
        chunks = conn.execute(text("SELECT COUNT(*) FROM chunks")).scalar_one()
        embedded = conn.execute(
            text("SELECT COUNT(*) FROM chunk_embeddings WHERE model = :m"), {"m": MODEL}
        ).scalar_one()
    return chunks, embedded


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--limit", type=int, help="embed at most N chunks this run")
    parser.add_argument("--status", action="store_true", help="report coverage and exit")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    engine = get_engine()
    init_db(engine)
    chunks, embedded = _status(engine)

    if args.status:
        print(f"{embedded:,} of {chunks:,} chunks embedded with {MODEL}")
        if embedded < chunks:
            pending = chunks - embedded
            mins = (pending / BATCH) * REQUEST_DELAY / 60
            print(f"{pending:,} pending — roughly {mins:.0f} min at the free-tier rate")
        return 0

    if not chunks:
        print("No chunks. Run scripts/rag_ingest.py first.")
        return 1

    pending = chunks - embedded
    if pending and not args.limit:
        print(f"embedding {pending:,} chunks — roughly "
              f"{(pending / BATCH) * REQUEST_DELAY / 60:.0f} min at the free-tier rate\n")

    started = time.time()
    out = embed_corpus(engine, limit=args.limit)
    elapsed = time.time() - started

    print(f"embedded {out['embedded']:,} chunk(s) in {elapsed:.0f}s")
    if out["pruned"]:
        print(f"  pruned {out['pruned']} vector(s) orphaned by a re-ingest")
    if out["stale"]:
        print(f"  dropped {out['stale']} vector(s) from a different model")

    chunks, embedded = _status(engine)
    print(f"\ncoverage: {embedded:,}/{chunks:,} chunks")
    if embedded < chunks:
        print("Run again to continue — already-embedded chunks are skipped.")
    else:
        print("Next: python scripts/eval_retrieval.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
