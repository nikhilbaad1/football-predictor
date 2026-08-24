"""Dense retrieval with Voyage embeddings.

The second half of the hybrid pipeline in PLAN section 4. ADR 0014 measured the
lexical half alone first so that whatever this adds is attributable rather than
assumed; that baseline is recall@5 = 11/13, MRR 0.750, and the comparison lives
in `scripts/eval_retrieval.py`.

**Vectors live in SQLite, not pgvector.** ADR 0005 says Postgres arrives when
pgvector is needed. Brute-force cosine over ~1,000 chunks is a single 1000x1024
matrix multiply — well under a millisecond — so the need has not arrived. Moving
first would be infrastructure bought on the assumption of a result that had not
been measured, which is the habit this project keeps arguing against.

**Documents and queries are embedded differently.** Voyage takes an `input_type`
and produces asymmetric vectors: "document" for the corpus, "query" for what the
user typed. Embedding both the same way is a quiet accuracy loss, not an error.
"""

from __future__ import annotations

import logging
import time

import numpy as np
from sqlalchemy import Engine, text

from fpp.db import get_engine

log = logging.getLogger(__name__)

MODEL = "voyage-3-large"
DIM = 1024
DTYPE = np.float32

# Sized for Voyage's *unbilled* free tier: 3 requests/min and 10,000 tokens/min
# until a payment method is on file. At ~245 tokens per chunk, 32 chunks is
# ~7.8k tokens, which fits one request inside the token cap; the delay then
# keeps the per-minute total under it too. Adding a payment method raises the
# limits without costing anything — the 200M free tokens still apply — at which
# point BATCH can go up and REQUEST_DELAY down to nearly zero.
BATCH = 32
REQUEST_DELAY = 25.0
MAX_RETRIES = 6


def _client(client=None):
    if client is not None:
        return client
    import voyageai  # imported lazily: nothing else needs the key

    from fpp import config  # noqa: F401  — loads .env before the SDK reads it

    return voyageai.Client()


def _embed_batch(client, batch: list[str], model: str, input_type: str):
    """One request, retried through rate limiting.

    The free tier answers a burst with RateLimitError rather than a header, so
    the backoff is exponential from the delay we already pace at. Failing the
    whole corpus because request 40 of 82 arrived a second early would be a poor
    trade for code this simple.
    """
    for attempt in range(MAX_RETRIES):
        try:
            return client.embed(batch, model=model, input_type=input_type)
        except Exception as exc:  # noqa: BLE001 - vendor error types vary by version
            if "rate limit" not in str(exc).lower() or attempt == MAX_RETRIES - 1:
                raise
            wait = REQUEST_DELAY * (2 ** attempt)
            log.warning("rate limited; waiting %.0fs (attempt %s)", wait, attempt + 1)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def embed_texts(
    texts: list[str],
    input_type: str,
    client=None,
    model: str = MODEL,
    delay: float = 0.0,
) -> np.ndarray:
    """Embed a list of texts, L2-normalised so cosine is a dot product.

    `input_type` must be "document" or "query" — see the module docstring.
    `delay` paces multi-batch corpus runs; a single query needs none.
    """
    if input_type not in {"document", "query"}:
        raise ValueError(f"input_type must be 'document' or 'query', got {input_type!r}")
    if not texts:
        return np.zeros((0, DIM), dtype=DTYPE)

    client = _client(client)
    vectors: list[list[float]] = []
    for start in range(0, len(texts), BATCH):
        if start and delay:
            time.sleep(delay)
        batch = texts[start:start + BATCH]
        result = _embed_batch(client, batch, model, input_type)
        vectors.extend(result.embeddings)
        log.info("embedded %s/%s", min(start + BATCH, len(texts)), len(texts))

    arr = np.asarray(vectors, dtype=DTYPE)
    # Normalising once here means every later comparison is a plain dot product
    # and no ranking depends on remembering to normalise at query time.
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.clip(norms, 1e-12, None)


def embed_corpus(
    engine: Engine | None = None,
    client=None,
    model: str = MODEL,
    limit: int | None = None,
) -> dict[str, int]:
    """Embed chunks that have no vector yet. Safe to re-run.

    Re-ingesting an article deletes and recreates its chunks with new ids, which
    leaves the old vectors orphaned. Those are pruned first, so a stale vector
    can never be joined back onto unrelated text.
    """
    engine = engine or get_engine()

    with engine.begin() as conn:
        pruned = conn.execute(
            text("DELETE FROM chunk_embeddings WHERE chunk_id NOT IN (SELECT id FROM chunks)")
        ).rowcount or 0
        # A model change invalidates everything: two models' vectors are not
        # comparable, and mixing them silently degrades every ranking.
        stale = conn.execute(
            text("DELETE FROM chunk_embeddings WHERE model <> :m"), {"m": model}
        ).rowcount or 0

    sql = text(
        """
        SELECT c.id, c.text FROM chunks c
        LEFT JOIN chunk_embeddings e ON e.chunk_id = c.id
        WHERE e.chunk_id IS NULL
        ORDER BY c.id
        """ + (f" LIMIT {int(limit)}" if limit else "")
    )
    with engine.connect() as conn:
        pending = conn.execute(sql).all()

    if not pending:
        return {"embedded": 0, "pruned": pruned, "stale": stale}

    client = _client(client)
    embedded = 0

    # Persist each batch as it arrives rather than accumulating and writing once
    # at the end. On the free tier a full corpus is a quarter of an hour of
    # paced requests, and losing all of it to a failure on the last one would be
    # a bad trade for a slightly tidier loop. A re-run resumes from here.
    for start in range(0, len(pending), BATCH):
        if start:
            time.sleep(REQUEST_DELAY)
        batch = pending[start:start + BATCH]
        vectors = embed_texts([row[1] for row in batch], "document", client, model)

        with engine.begin() as conn:
            for (chunk_id, _), vector in zip(batch, vectors, strict=True):
                conn.execute(
                    text(
                        """
                        INSERT INTO chunk_embeddings (chunk_id, model, dim, vector, created_at)
                        VALUES (:c, :m, :d, :v, CURRENT_TIMESTAMP)
                        ON CONFLICT (chunk_id) DO UPDATE SET
                            model = excluded.model, dim = excluded.dim,
                            vector = excluded.vector, created_at = CURRENT_TIMESTAMP
                        """
                    ),
                    {"c": chunk_id, "m": model, "d": len(vector),
                     "v": vector.astype(DTYPE).tobytes()},
                )
        embedded += len(batch)
        log.info("persisted %s/%s", embedded, len(pending))

    return {"embedded": embedded, "pruned": pruned, "stale": stale}


def load_vectors(
    engine: Engine | None = None,
    team: str | None = None,
    season: str | None = None,
    model: str = MODEL,
) -> tuple[list[dict], np.ndarray]:
    """Chunks with vectors, plus the matrix. Metadata filter applies first."""
    engine = engine or get_engine()
    where = ["e.model = :model"]
    params: dict = {"model": model}
    if team:
        where.append("d.team = :team")
        params["team"] = team
    if season:
        where.append("d.season = :season")
        params["season"] = season

    sql = text(
        f"""
        SELECT c.id, c.text, c.heading, c.n_words,
               d.title, d.url, d.team, d.season, d.license, e.vector
        FROM chunk_embeddings e
        JOIN chunks c ON c.id = e.chunk_id
        JOIN documents d ON d.id = c.document_id
        WHERE {' AND '.join(where)}
        ORDER BY c.id
        """
    )
    with engine.connect() as conn:
        rows = [dict(r) for r in conn.execute(sql, params).mappings().all()]

    if not rows:
        return [], np.zeros((0, DIM), dtype=DTYPE)

    matrix = np.vstack([np.frombuffer(r.pop("vector"), dtype=DTYPE) for r in rows])
    return rows, matrix


def dense_search(
    query: str,
    engine: Engine | None = None,
    team: str | None = None,
    season: str | None = None,
    limit: int = 5,
    client=None,
    model: str = MODEL,
    cached: tuple[list[dict], np.ndarray] | None = None,
    query_vector: np.ndarray | None = None,
) -> list[dict]:
    """Rank chunks by cosine similarity to the query embedding.

    `query_vector` lets a caller supply an already-embedded query. That is not
    a micro-optimisation: the unbilled free tier allows three requests a minute,
    so an eval over thirteen questions in two modes would spend nine minutes
    waiting on twenty-six one-item requests it could have made as one.
    """
    rows, matrix = cached if cached is not None else load_vectors(engine, team, season, model)
    if not rows:
        return []

    q = query_vector if query_vector is not None else embed_texts(
        [query], "query", client, model
    )[0]
    scores = matrix @ q          # both sides normalised, so this is cosine
    order = np.argsort(scores)[::-1][:limit]

    return [
        {**rows[int(i)], "score": round(float(scores[i]), 4), "rank": rank}
        for rank, i in enumerate(order, start=1)
    ]
