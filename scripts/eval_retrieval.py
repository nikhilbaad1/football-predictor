#!/usr/bin/env python3
"""Score retrieval on a hand-written question set: lexical, dense, and fused.

    python scripts/eval_retrieval.py                 # all three modes
    python scripts/eval_retrieval.py --mode lexical  # free, no API calls
    python scripts/eval_retrieval.py -k 3 --verbose

Lexical is free. Dense and hybrid embed each question once — all of them in a
single request, because the unbilled Voyage tier allows three requests a minute
and one call per question per mode would spend the run waiting.

ADR 0014 measured lexical alone first so that whatever dense adds is
attributable rather than assumed. The lexical baseline is recall@5 = 11/13,
MRR 0.750. A mode that does not beat that is reported as not beating it — a
negative result, honestly measured, is the point of having the harness.

A case passes when a chunk from the expected document appears in the top k and,
where given, contains `must_contain`. On a failure the script checks whether the
expected text exists in the expected document at all, and reports a BROKEN case
rather than scoring a bad expectation as a retrieval miss.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fpp.rag import load_chunks, search  # noqa: E402
from fpp.rag.dense import dense_search, embed_texts, load_vectors  # noqa: E402
from fpp.rag.hybrid import hybrid_search  # noqa: E402

MODES = ("lexical", "dense", "hybrid")


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
# which is where dense retrieval is supposed to earn its place.
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
         "2023–24 Premier League", "record",
         note="lexical loses this to club 'Record goalscorers' sections"),
    Case("first team relegated to the Championship",
         "2023–24 Premier League", "relegated"),
    Case("manager sacked after one win in thirteen games",
         "2023–24 Premier League", "sack",
         note="lexical loses this to club-history managerial sackings"),
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


def rank_of_expected(results: list[dict], case: Case) -> int | None:
    for r in results:
        if r["title"] != case.doc:
            continue
        if case.must_contain and case.must_contain.lower() not in r["text"].lower():
            continue
        return r["rank"]
    return None


def diagnose(case: Case, corpus: list[dict]) -> str | None:
    """Is the case itself sound? Returns a reason if it is broken."""
    pool = [c for c in corpus if c["title"] == case.doc]
    if not pool:
        return f"no document titled {case.doc!r}"
    if case.must_contain and not any(
        case.must_contain.lower() in c["text"].lower() for c in pool
    ):
        return f"{case.doc!r} never contains {case.must_contain!r}"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("-k", type=int, default=5, help="cutoff for recall@k")
    parser.add_argument("--mode", choices=[*MODES, "all"], default="all")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    corpus = load_chunks()
    if not corpus:
        print("Corpus is empty. Run scripts/rag_ingest.py first.")
        return 1

    modes = list(MODES) if args.mode == "all" else [args.mode]

    if set(modes) & {"dense", "hybrid"}:
        rows, _ = load_vectors()
        if not rows:
            print("No embeddings. Run scripts/rag_embed.py first.")
            return 1
        if len(rows) < len(corpus):
            print(f"WARNING: {len(rows)} of {len(corpus)} chunks embedded — "
                  f"dense results are over a partial corpus.\n")

    # Broken cases are excluded from every mode, so the comparison stays fair.
    broken = {c.query: r for c in CASES if (r := diagnose(c, corpus))}
    scored = [c for c in CASES if c.query not in broken]

    print(f"{len(scored)} cases · {len(corpus)} chunks · recall@{args.k}\n")

    # One request for every question, reused by dense and hybrid.
    vectors = {}
    if set(modes) & {"dense", "hybrid"}:
        embedded = embed_texts([c.query for c in scored], "query")
        vectors = {c.query: v for c, v in zip(scored, embedded, strict=True)}

    results: dict[str, dict] = {}
    per_case: dict[str, dict[str, int | None]] = {c.query: {} for c in scored}

    for mode in modes:
        hits = 0
        reciprocal = 0.0
        for case in scored:
            common = dict(team=case.team, season=case.season, limit=args.k)
            if mode == "lexical":
                found = search(case.query, **common)
            elif mode == "dense":
                found = dense_search(case.query, query_vector=vectors[case.query], **common)
            else:
                found = hybrid_search(case.query, query_vector=vectors[case.query], **common)

            rank = rank_of_expected(found, case)
            per_case[case.query][mode] = rank
            if rank:
                hits += 1
                reciprocal += 1 / rank
        results[mode] = {
            "recall": hits / len(scored),
            "hits": hits,
            "mrr": reciprocal / len(scored),
        }

    width = max(len(c.query) for c in scored)
    print(f"  {'query':{width}}  " + "  ".join(f"{m:>7}" for m in modes))
    print(f"  {'-' * width}  " + "  ".join("-" * 7 for _ in modes))
    for case in scored:
        cells = []
        for mode in modes:
            rank = per_case[case.query][mode]
            cells.append(f"{('@' + str(rank)) if rank else '—':>7}")
        print(f"  {case.query:{width}}  " + "  ".join(cells))
        if case.note and args.verbose:
            print(f"      {case.note}")

    print(f"\n{'=' * (width + 4 + 9 * len(modes))}")
    print(f"  {'recall@' + str(args.k):{width}}  " +
          "  ".join(f"{results[m]['hits']}/{len(scored):<5}" for m in modes))
    print(f"  {'MRR':{width}}  " + "  ".join(f"{results[m]['mrr']:7.3f}" for m in modes))

    if "lexical" in results and len(modes) > 1:
        base = results["lexical"]
        print()
        for mode in modes:
            if mode == "lexical":
                continue
            d_recall = results[mode]["hits"] - base["hits"]
            d_mrr = results[mode]["mrr"] - base["mrr"]
            verdict = ("beats" if (d_recall > 0 or (d_recall == 0 and d_mrr > 1e-9))
                       else "does not beat")
            print(f"  {mode} {verdict} lexical: "
                  f"recall {d_recall:+d}, MRR {d_mrr:+.3f}")

    if broken:
        print(f"\n{len(broken)} BROKEN case(s) — excluded from every mode:")
        for query, reason in broken.items():
            print(f"  {query!r}: {reason}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
