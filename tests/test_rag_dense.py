"""Dense retrieval and fusion tests.

No API calls: CI has no Voyage key, and a test whose result depends on an
embedding model measures the model rather than the code. Scoring retrieval is a
separate, deliberate job — `scripts/eval_retrieval.py`.

Reciprocal Rank Fusion is checked against values computed by hand from the
formula, like BM25 in test_rag.py and the Elo and Dixon-Coles maths.
"""

from __future__ import annotations

import numpy as np
import pytest
from sqlalchemy import create_engine, text

from fpp.db import init_db
from fpp.rag.dense import DIM, MODEL, dense_search, embed_corpus, embed_texts, load_vectors
from fpp.rag.hybrid import RRF_K, hybrid_search, reciprocal_rank_fusion
from fpp.rag.wikipedia import store_document


class FakeEmbeddings:
    def __init__(self, vectors):
        self.embeddings = vectors
        self.total_tokens = 10


class FakeVoyage:
    """Returns a deterministic vector per text, and records how it was called."""

    def __init__(self, mapping=None):
        self.mapping = mapping or {}
        self.calls = []

    def embed(self, texts, model=None, input_type=None):
        self.calls.append({"texts": list(texts), "model": model, "input_type": input_type})
        out = []
        for t in texts:
            if t in self.mapping:
                out.append(list(self.mapping[t]))
            else:
                # Arbitrary but stable: seeded by the text, so the same string
                # always embeds the same way.
                rng = np.random.default_rng(abs(hash(t)) % (2**32))
                out.append(list(rng.normal(size=DIM)))
        return FakeEmbeddings(out)


def unit(*components) -> np.ndarray:
    """A DIM-length vector with the given leading components, normalised."""
    v = np.zeros(DIM)
    v[: len(components)] = components
    return v / np.linalg.norm(v)


class TestEmbedTexts:
    def test_rejects_an_invalid_input_type(self):
        """Voyage embeds documents and queries asymmetrically. Passing the wrong
        one is a quiet accuracy loss, so it is refused rather than defaulted."""
        with pytest.raises(ValueError, match="input_type"):
            embed_texts(["x"], "passage", client=FakeVoyage())

    @pytest.mark.parametrize("input_type", ["document", "query"])
    def test_input_type_is_forwarded(self, input_type):
        client = FakeVoyage()
        embed_texts(["x"], input_type, client=client)
        assert client.calls[0]["input_type"] == input_type

    def test_vectors_come_back_normalised(self):
        """Normalising once at write time makes every later comparison a plain
        dot product, so no ranking depends on remembering to normalise."""
        client = FakeVoyage({"a": [3.0] + [4.0] + [0.0] * (DIM - 2)})
        out = embed_texts(["a"], "document", client=client)
        assert np.linalg.norm(out[0]) == pytest.approx(1.0, abs=1e-6)

    def test_empty_input_makes_no_request(self):
        client = FakeVoyage()
        out = embed_texts([], "document", client=client)
        assert out.shape == (0, DIM)
        assert client.calls == []

    def test_long_input_is_batched(self):
        from fpp.rag.dense import BATCH
        client = FakeVoyage()
        embed_texts([f"t{i}" for i in range(BATCH + 5)], "document", client=client)
        assert len(client.calls) == 2


class TestReciprocalRankFusion:
    """k = 60, so rank 1 contributes 1/61 and rank 2 contributes 1/62."""

    def test_agreement_beats_enthusiasm(self):
        """The property fusion exists for.

        A is only second in both rankings; X and Y are each first in one. A
        wins, because two rankers agreeing is better evidence than one ranker
        being certain. With k = 60 the gap between first and second is small
        enough for that to hold — which is the reason for the constant.
        """
        lexical = [{"id": "X"}, {"id": "A"}]
        dense = [{"id": "Y"}, {"id": "A"}]
        fused = reciprocal_rank_fusion([lexical, dense], limit=3)
        assert fused[0]["id"] == "A"
        # Tolerance is 1e-6 because rrf_score is rounded to six places on the
        # way out; ordering happens before that, on the full value.
        assert fused[0]["rrf_score"] == pytest.approx(2 / 62, abs=1e-6)
        assert fused[1]["rrf_score"] == pytest.approx(1 / 61, abs=1e-6)

    def test_score_matches_the_formula(self):
        fused = reciprocal_rank_fusion([[{"id": "A"}, {"id": "B"}]], limit=2)
        assert fused[0]["rrf_score"] == pytest.approx(1 / (RRF_K + 1), abs=1e-6)
        assert fused[1]["rrf_score"] == pytest.approx(1 / (RRF_K + 2), abs=1e-6)

    def test_a_document_in_only_one_ranking_is_not_penalised(self):
        """It forfeits the missing ranker's contribution and nothing more."""
        fused = reciprocal_rank_fusion([[{"id": "A"}], [{"id": "B"}]], limit=2)
        assert {r["id"] for r in fused} == {"A", "B"}
        assert fused[0]["rrf_score"] == fused[1]["rrf_score"]

    def test_provenance_is_reported(self):
        """A fused number nobody can account for is worse than two rankings."""
        fused = reciprocal_rank_fusion([[{"id": "A"}], [{"id": "A"}]], limit=1)
        assert fused[0]["found_by"] == 2
        assert fused[0]["source_ranks"] == [1, 1]

    def test_ranks_are_renumbered_from_one(self):
        fused = reciprocal_rank_fusion([[{"id": "A"}, {"id": "B"}]], limit=2)
        assert [r["rank"] for r in fused] == [1, 2]

    def test_empty_input_is_survivable(self):
        assert reciprocal_rank_fusion([[], []]) == []


LONG_A = ("Arsenal play at the Emirates Stadium which has a capacity of about "
          "sixty thousand supporters and which opened in two thousand and six "
          "after the club moved from Highbury, their home for many decades.")
LONG_B = ("Chelsea play at Stamford Bridge in west London and the ground has "
          "been their home since eighteen ninety five without interruption, "
          "making it one of the oldest continuously used grounds in England.")


def _page(title, body):
    return {"title": title, "text": body, "url": f"https://en.wikipedia.org/wiki/{title}"}


@pytest.fixture
def corpus(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path / 'dense.db'}", future=True)
    init_db(eng)
    store_document(_page("Arsenal F.C.", LONG_A), eng, team="Arsenal")
    store_document(_page("Chelsea F.C.", LONG_B), eng, team="Chelsea")
    return eng


def _texts(engine):
    with engine.connect() as c:
        return [r[0] for r in c.execute(text("SELECT text FROM chunks ORDER BY id"))]


class TestEmbedCorpus:
    def test_embeds_every_chunk_once(self, corpus):
        out = embed_corpus(corpus, client=FakeVoyage())
        assert out["embedded"] == 2
        with corpus.connect() as c:
            assert c.execute(text("SELECT COUNT(*) FROM chunk_embeddings")).scalar_one() == 2

    def test_rerunning_embeds_nothing_new(self, corpus):
        embed_corpus(corpus, client=FakeVoyage())
        client = FakeVoyage()
        assert embed_corpus(corpus, client=client)["embedded"] == 0
        assert client.calls == [], "paid for embeddings that already existed"

    def test_orphaned_vectors_are_pruned(self, corpus):
        """Re-ingesting an article recreates its chunks with new ids. A vector
        left behind could otherwise be joined onto unrelated text."""
        embed_corpus(corpus, client=FakeVoyage())
        store_document(_page("Arsenal F.C.", LONG_B), corpus, team="Arsenal")
        out = embed_corpus(corpus, client=FakeVoyage())
        assert out["pruned"] == 1
        with corpus.connect() as c:
            orphans = c.execute(
                text("""SELECT COUNT(*) FROM chunk_embeddings e
                        LEFT JOIN chunks c ON c.id = e.chunk_id
                        WHERE c.id IS NULL""")
            ).scalar_one()
        assert orphans == 0

    def test_changing_model_invalidates_everything(self, corpus):
        """Two models' vectors are not comparable; mixing them would degrade
        every ranking without failing anything."""
        embed_corpus(corpus, client=FakeVoyage(), model="old-model")
        out = embed_corpus(corpus, client=FakeVoyage(), model=MODEL)
        assert out["stale"] == 2
        assert out["embedded"] == 2
        with corpus.connect() as c:
            models = {r[0] for r in c.execute(text("SELECT DISTINCT model FROM chunk_embeddings"))}
        assert models == {MODEL}


class TestDenseSearch:
    def test_ranks_by_cosine_similarity(self, corpus):
        texts = _texts(corpus)
        # Arsenal's chunk points along axis 0, Chelsea's along axis 1.
        client = FakeVoyage({texts[0]: unit(1, 0), texts[1]: unit(0, 1)})
        embed_corpus(corpus, client=client)

        hits = dense_search("anything", engine=corpus, limit=2,
                            query_vector=unit(1, 0))
        assert hits[0]["title"] == "Arsenal F.C."
        assert hits[0]["score"] == pytest.approx(1.0, abs=1e-4)
        assert hits[1]["score"] == pytest.approx(0.0, abs=1e-4)

    def test_a_supplied_query_vector_skips_the_api(self, corpus):
        embed_corpus(corpus, client=FakeVoyage())
        client = FakeVoyage()
        dense_search("q", engine=corpus, query_vector=unit(1, 0), client=client)
        assert client.calls == [], "embedded a query that was already supplied"

    def test_metadata_filter_restricts_the_pool(self, corpus):
        embed_corpus(corpus, client=FakeVoyage())
        hits = dense_search("q", engine=corpus, team="Chelsea", query_vector=unit(1, 0))
        assert hits and all(h["team"] == "Chelsea" for h in hits)

    def test_no_embeddings_returns_nothing(self, corpus):
        assert dense_search("q", engine=corpus, query_vector=unit(1, 0)) == []

    def test_load_vectors_shape_matches_rows(self, corpus):
        embed_corpus(corpus, client=FakeVoyage())
        rows, matrix = load_vectors(corpus)
        assert matrix.shape == (len(rows), DIM)


class TestHybridSearch:
    def test_combines_both_rankers(self, corpus):
        texts = _texts(corpus)
        client = FakeVoyage({texts[0]: unit(1, 0), texts[1]: unit(0, 1)})
        embed_corpus(corpus, client=client)

        hits = hybrid_search("Emirates Stadium capacity", engine=corpus,
                             limit=2, query_vector=unit(1, 0))
        assert hits
        assert hits[0]["title"] == "Arsenal F.C."
        # Lexical and dense both surfaced it, and the result says so.
        assert hits[0]["found_by"] == 2
