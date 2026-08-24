"""Reciprocal Rank Fusion of lexical and dense results.

PLAN section 4 specifies RRF, and the reason is worth stating: BM25 scores and
cosine similarities are not on the same scale and have no fixed range, so any
weighted sum of the two raw scores is arbitrary. RRF ignores the scores entirely
and fuses on *rank*, which is the only thing the two rankers report comparably.

    score(d) = sum over rankers of 1 / (k + rank(d))

k = 60 is the constant from the original Cormack et al. paper and the usual
default. It damps the gap between the top few positions: with k = 60 the first
and second places differ by about 1.6%, so a document both rankers place highly
beats one that either ranker alone loves. That is the whole point of fusing —
agreement should outrank enthusiasm.
"""

from __future__ import annotations

from typing import Any

RRF_K = 60


def reciprocal_rank_fusion(
    rankings: list[list[dict[str, Any]]],
    k: int = RRF_K,
    limit: int = 5,
    id_key: str = "id",
) -> list[dict[str, Any]]:
    """Fuse several ranked lists into one.

    Each input is a list of result dicts already ordered best-first. A document
    missing from a ranking simply contributes nothing for it, which is the
    behaviour that matters: a chunk no lexical term matched is not penalised
    beyond forfeiting that ranker's contribution.
    """
    fused: dict[Any, dict[str, Any]] = {}
    contributions: dict[Any, list[int]] = {}

    for ranking in rankings:
        for rank, row in enumerate(ranking, start=1):
            key = row[id_key]
            fused.setdefault(key, dict(row))
            fused[key].setdefault("rrf_score", 0.0)
            fused[key]["rrf_score"] += 1.0 / (k + rank)
            contributions.setdefault(key, []).append(rank)

    ordered = sorted(fused.values(), key=lambda r: r["rrf_score"], reverse=True)

    out = []
    for rank, row in enumerate(ordered[:limit], start=1):
        row["rank"] = rank
        row["rrf_score"] = round(row["rrf_score"], 6)
        # How many rankers found it, and where. This is what makes a fused
        # result explainable rather than a number nobody can account for.
        row["found_by"] = len(contributions[row[id_key]])
        row["source_ranks"] = contributions[row[id_key]]
        out.append(row)
    return out


def hybrid_search(
    query: str,
    engine=None,
    team: str | None = None,
    season: str | None = None,
    limit: int = 5,
    depth: int = 20,
    client=None,
    query_vector=None,
) -> list[dict[str, Any]]:
    """Lexical and dense, fused.

    Each ranker is asked for `depth` results rather than `limit`, because fusion
    can only promote what it is given: a chunk ranked 8th by one ranker and 2nd
    by the other should win, and it cannot if the lists were truncated at 5.

    `query_vector` passes an already-embedded query straight through to the
    dense side, so a caller running several modes pays for one embedding.
    """
    from fpp.rag.dense import dense_search
    from fpp.rag.lexical import search as lexical_search

    lexical = lexical_search(query, engine=engine, team=team, season=season, limit=depth)
    dense = dense_search(query, engine=engine, team=team, season=season,
                         limit=depth, client=client, query_vector=query_vector)
    return reciprocal_rank_fusion([lexical, dense], limit=limit)
