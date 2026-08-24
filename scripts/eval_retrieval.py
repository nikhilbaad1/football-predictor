#!/usr/bin/env python3
"""Score lexical retrieval on a hand-written question set.

    python scripts/eval_retrieval.py
    python scripts/eval_retrieval.py --k 3 --verbose

Free — no API calls. This is the number the dense half has to beat later. PLAN
section 4 claims BM25 is "the highest-impact addition to a pure vector
pipeline"; that is a claim about lexical matching's contribution, and it is only
checkable against a measurement of lexical alone. Same discipline as ADR 0008.

A case passes when a chunk from the expected document appears in the top k and,
where given, contains `must_contain`. On a failure the script checks whether the
expected document contains that text at all, so a wrong expectation is reported
as a broken case rather than counted as a retrieval miss — an eval that quietly
scores its own mistakes as model failures is worse than no eval.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fpp.rag import load_chunks, search  # noqa: E402


@dataclass
class Case:
    query: str
    doc: str                      # expected document title
    must_contain: str | None = None
    team: str | None = None
    season: str | None = None
    note: str = ""


# Deliberately mixed. Some cases share vocabulary with the passage, which is
# what lexical search is good at; others are paraphrases with almost no overlap,
# which is where it should struggle and dense retrieval should earn its place.
CASES = [
    # -- keyword overlap: lexical should handle these --
    Case("stadium capacity all-seater", "Arsenal F.C.", "capacity", team="Arsenal"),
    Case("team deducted points for breaching financial rules",
         "2023–24 Premier League", "deducted"),
    Case("player suffered a cardiac arrest and the match was abandoned",
         "2023–24 Premier League", "cardiac"),
    Case("VAR failed to intervene on a disallowed goal",
         "2023–24 Premier League", "VAR"),
    Case("record number of goals scored in a season",
         "2023–24 Premier League", "record"),
    Case("first team relegated to the Championship",
         "2023–24 Premier League", "relegated"),
    Case("manager sacked after one win in thirteen games",
         "2023–24 Premier League", "sack"),
    Case("when was the club founded", "Hull City A.F.C.", "1904", team="Hull"),
    Case("City Football Group ownership stakes in other clubs",
         "Manchester City F.C.", "City Football Group", team="Manchester City"),

    # -- paraphrase: little or no shared vocabulary with the passage --
    Case("where do they play their home games", "Newcastle United F.C.",
         "St. James", team="Newcastle United",
         note="'home games' vs 'stadium/venue' — no keyword bridge"),
    Case("who are their biggest rivals", "Liverpool F.C.", None, team="Liverpool",
         note="'biggest rivals' vs 'rivalry/derby'"),
    Case("what do the supporters call the team", "Everton F.C.", None, team="Everton",
         note="'supporters call' vs 'nickname'"),
    Case("how much money does the club make", "Manchester United F.C.", None,
         team="Manchester United", note="'make money' vs 'revenue/finances'"),
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-k", type=int, default=5, help="cutoff for recall@k")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    corpus = load_chunks()
    if not corpus:
        print("Corpus is empty. Run scripts/rag_ingest.py first.")
        return 1

    print(f"{len(CASES)} cases · {len(corpus)} chunks · recall@{args.k}\n")

    hits = 0
    reciprocal = 0.0
    broken: list[str] = []

    for case in CASES:
        results = search(case.query, team=case.team, season=case.season, limit=args.k)

        rank = None
        for r in results:
            if r["title"] != case.doc:
                continue
            if case.must_contain and case.must_contain.lower() not in r["text"].lower():
                continue
            rank = r["rank"]
            break

        if rank:
            hits += 1
            reciprocal += 1 / rank
            mark = f"ok  @{rank}"
        else:
            # Is the case itself sound? If the expected text is not in the
            # expected document, this is a bug in the eval, not a miss.
            pool = [c for c in corpus if c["title"] == case.doc]
            if not pool:
                broken.append(f"{case.query!r}: no document titled {case.doc!r}")
                mark = "BROKEN"
            elif case.must_contain and not any(
                case.must_contain.lower() in c["text"].lower() for c in pool
            ):
                broken.append(
                    f"{case.query!r}: {case.doc!r} never contains {case.must_contain!r}"
                )
                mark = "BROKEN"
            else:
                mark = "miss"

        print(f"  {mark:7s} {case.query[:52]:52s} -> {case.doc[:26]}")
        if case.note and (mark.startswith("miss") or args.verbose):
            print(f"          {case.note}")

    n = len(CASES)
    scored = n - len(broken)
    print(f"\n{'=' * 66}")
    if not scored:
        print("no scorable cases")
        return 1
    print(f"recall@{args.k}   {hits}/{scored}  ({hits / scored:.0%})")
    print(f"MRR         {reciprocal / scored:.3f}")

    # Scale reference: picking k chunks at random from the whole corpus. The
    # filtered cases are easier than this suggests, so it is a floor, not a rival.
    print(f"random@{args.k}   ~{args.k / len(corpus):.1%}  (floor, for scale)")

    if broken:
        print(f"\n{len(broken)} BROKEN case(s) — excluded from the score:")
        for b in broken:
            print(f"  {b}")

    print("\nThis is the lexical-only number. Dense retrieval is added next and")
    print("has to beat it on this same set, or it does not go in (ADR 0014).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
