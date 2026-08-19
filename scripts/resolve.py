#!/usr/bin/env python3
"""Run the entity-resolution agent over names no rule would commit to.

    python scripts/resolve.py            # propose decisions, write nothing
    python scripts/resolve.py --apply    # commit them after you have read them

Proposing is the default on purpose. A wrong match fuses two clubs' histories
and produces rows that all look valid, so the model's judgement is reviewed
before it reaches the database — see ADR 0013. Nothing here writes without
--apply, and an "uncertain" decision is never applied at all.

Each name costs about a cent. `scripts/eval_resolver.py` scores the agent
against the deterministic baseline.
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
from fpp.ingest.fpl import apply_resolution, pending_resolutions  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--apply", action="store_true",
                        help="write the proposed decisions to the database")
    parser.add_argument("--limit", type=int, help="only handle the first N names")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    engine = get_engine()
    pending = pending_resolutions(engine)
    if args.limit:
        pending = pending[: args.limit]

    if not pending:
        print("Nothing pending — every source name has been resolved.")
        return 0

    with engine.connect() as conn:
        known = {r[0] for r in conn.execute(text("SELECT name FROM teams"))}

    print(f"{len(pending)} name(s) to decide · {len(known)} known clubs\n")

    cost = 0.0
    applied = skipped = 0

    for row in pending:
        raw = row["raw_name"]
        result = resolve_with_agent(raw, known)
        cost += result.cost_usd

        target = f" -> {result.resolved_to}" if result.resolved_to else ""
        print(f"  {raw!r}")
        print(f"    {result.decision}{target}  (confidence {result.confidence:.2f})")
        print(f"    {result.reasoning}")
        if result.validation_note:
            print(f"    validation: {result.validation_note}")

        if not args.apply:
            continue

        if result.decision == "uncertain":
            print("    not applied — uncertain decisions go to a human")
            skipped += 1
            continue

        try:
            name = apply_resolution(
                raw_name=raw,
                decision=result.decision,
                resolved_to=result.resolved_to,
                engine=engine,
                source=row["source"],
                entity_type=row["entity_type"],
                method=f"agent:{result.agent_version}",
                confidence=result.confidence,
                rationale=result.reasoning,
            )
            verb = "linked to" if result.decision == "matched" else "created as"
            print(f"    applied — {verb} {name!r}")
            applied += 1
        except ValueError as exc:
            print(f"    NOT applied — {exc}")
            skipped += 1
        print()

    print(f"\ncost ${cost:.4f}")
    if args.apply:
        print(f"applied {applied} · skipped {skipped}")
        if applied:
            print("\nRe-run python scripts/fpl.py to ingest any squads that were "
                  "skipped while their club was unresolved.")
    else:
        print("\nNothing written. Re-run with --apply to commit these decisions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
