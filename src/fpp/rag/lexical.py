"""BM25 retrieval over the text corpus.

The lexical half of the hybrid pipeline in PLAN section 4, built and measured
first. That ordering is the point: PLAN calls BM25 "the highest-impact addition
to a pure vector pipeline", which is a claim about what lexical matching
contributes, and the only way to know what dense retrieval adds on top is to
have a number for lexical alone. Same discipline as ADR 0008 — establish the
baseline, then make the fancier thing beat it.

BM25 is written out here rather than imported. It is a closed-form formula, so
the tests check it against values computed by hand, exactly as the Dixon-Coles
tau correction and the Elo update are. Importing it would leave a component
nobody in this project can explain under questioning.

No stemming. PLAN section 4 wants lexical search precisely for "exact terms like
player and manager names", and a stemmer is most aggressive on the proper nouns
that matter most here.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from sqlalchemy import Engine, text

from fpp.db import get_engine

K1 = 1.5   # term-frequency saturation
B = 0.75   # length normalisation

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(raw: str) -> list[str]:
    return _TOKEN.findall(raw.lower())


@dataclass
class BM25Index:
    """Okapi BM25 with the Lucene IDF variant.

        score(D,Q) = sum_q IDF(q) * f(q,D)*(k1+1)
                                  / (f(q,D) + k1*(1 - b + b*|D|/avgdl))

        IDF(q)     = ln(1 + (N - n(q) + 0.5) / (n(q) + 0.5))

    The +1 inside the log is what keeps IDF positive for a term appearing in
    most documents. The textbook form goes negative there, which lets a common
    word actively push a document *down* the ranking — for a corpus where every
    document says "football" and "season", that is a real effect, not a corner
    case.
    """

    k1: float = K1
    b: float = B
    corpus: list[list[str]] = field(default_factory=list)
    doc_freq: Counter = field(default_factory=Counter)
    doc_len: np.ndarray = field(default_factory=lambda: np.zeros(0))
    avg_len: float = 0.0

    def fit(self, documents: list[str]) -> BM25Index:
        self.corpus = [tokenize(d) for d in documents]
        self.doc_len = np.array([len(d) for d in self.corpus], dtype=float)
        self.avg_len = float(self.doc_len.mean()) if len(self.doc_len) else 0.0
        self.doc_freq = Counter()
        for tokens in self.corpus:
            self.doc_freq.update(set(tokens))
        return self

    def idf(self, term: str) -> float:
        n = self.doc_freq.get(term, 0)
        return math.log(1 + (len(self.corpus) - n + 0.5) / (n + 0.5))

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.corpus))
        if not self.corpus:
            return out
        for term in tokenize(query):
            if term not in self.doc_freq:
                continue  # unseen term contributes nothing to any document
            idf = self.idf(term)
            for i, tokens in enumerate(self.corpus):
                freq = tokens.count(term)
                if not freq:
                    continue
                norm = 1 - self.b + self.b * (self.doc_len[i] / self.avg_len)
                out[i] += idf * freq * (self.k1 + 1) / (freq + self.k1 * norm)
        return out


def load_chunks(
    engine: Engine | None = None,
    team: str | None = None,
    season: str | None = None,
) -> list[dict]:
    """Chunks, optionally narrowed by document metadata.

    Filtering before ranking is the first stage of the PLAN section 4 cascade.
    It is not an optimisation: "how did they do last season" scoped to the wrong
    club returns fluent, well-ranked, wrong text.
    """
    engine = engine or get_engine()
    where, params = [], {}
    if team:
        where.append("d.team = :team")
        params["team"] = team
    if season:
        where.append("d.season = :season")
        params["season"] = season
    clause = f"WHERE {' AND '.join(where)}" if where else ""

    sql = text(
        f"""
        SELECT c.id, c.text, c.heading, c.n_words,
               d.title, d.url, d.team, d.season, d.license
        FROM chunks c JOIN documents d ON d.id = c.document_id
        {clause}
        ORDER BY c.id
        """
    )
    with engine.connect() as conn:
        return [dict(r) for r in conn.execute(sql, params).mappings().all()]


def search(
    query: str,
    engine: Engine | None = None,
    team: str | None = None,
    season: str | None = None,
    limit: int = 5,
    chunks: list[dict] | None = None,
) -> list[dict]:
    """Rank chunks against a query. Returns the top `limit` with scores.

    The index is rebuilt per call. At a few hundred chunks that is milliseconds
    and it keeps the corpus and the index from ever disagreeing; it becomes
    worth caching when the corpus is large enough for that to show up in a
    measurement, and not before.
    """
    rows = chunks if chunks is not None else load_chunks(engine, team, season)
    if not rows:
        return []

    index = BM25Index().fit([r["text"] for r in rows])
    scores = index.scores(query)
    order = np.argsort(scores)[::-1][:limit]

    results = []
    for rank, i in enumerate(order, start=1):
        if scores[i] <= 0:
            break  # nothing matched; padding the list would invent relevance
        results.append({**rows[int(i)], "score": round(float(scores[i]), 4), "rank": rank})
    return results
