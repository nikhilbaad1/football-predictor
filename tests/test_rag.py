"""Retrieval tests.

BM25 is checked against values computed by hand from the formula, the way the
Elo update and the Dixon-Coles tau correction are. A snapshot test would only
prove the ranker still does what it did yesterday; these fail for a reason that
can be stated.

The other test that carries weight is TestSportCheck. "Hull F.C." is a rugby
league club, and it passed an earlier `"football club" in text` check because
English rugby league clubs are also named Football Club — putting 29 chunks of
rugby into the corpus under a Premier League team.
"""

from __future__ import annotations

import math

import pytest
from sqlalchemy import create_engine, text

from fpp.db import init_db
from fpp.rag.lexical import BM25Index, search, tokenize
from fpp.rag.wikipedia import (
    MAX_WORDS,
    MIN_WORDS,
    chunk_text,
    is_association_football_club,
    store_document,
)


class TestTokenizer:
    def test_lowercases_and_splits_on_punctuation(self):
        assert tokenize("Nott'm Forest, 2-1!") == ["nott", "m", "forest", "2", "1"]

    def test_keeps_digits(self):
        """Years and scorelines are real query terms."""
        assert "1904" in tokenize("founded in 1904")

    def test_empty_string_is_empty(self):
        assert tokenize("") == []


class TestBM25AgainstHandComputedValues:
    """Corpus: ["a b", "a c d"]. N=2, lengths [2,3], avgdl=2.5, k1=1.5, b=0.75."""

    @pytest.fixture
    def index(self):
        return BM25Index().fit(["a b", "a c d"])

    def test_idf_of_a_term_in_every_document(self, index):
        # ln(1 + (2 - 2 + 0.5) / (2 + 0.5)) = ln(1.2)
        assert index.idf("a") == pytest.approx(math.log(1.2), rel=1e-9)

    def test_idf_stays_positive_for_ubiquitous_terms(self, index):
        """The textbook IDF goes negative here, which lets a word every document
        contains push documents *down*. In this corpus 'football' and 'season'
        are in nearly everything, so that is a live effect, not a corner case."""
        assert index.idf("a") > 0

    def test_rarer_term_scores_higher(self, index):
        assert index.idf("c") > index.idf("a")
        # ln(1 + (2 - 1 + 0.5) / (1 + 0.5)) = ln(2)
        assert index.idf("c") == pytest.approx(math.log(2.0), rel=1e-9)

    def test_score_matches_the_formula(self, index):
        # doc0: f=1, |D|=2 -> norm = 1 - 0.75 + 0.75*(2/2.5) = 0.85
        #       0.1823216 * 1*2.5 / (1 + 1.5*0.85) = 0.2003533
        # doc1: f=1, |D|=3 -> norm = 1 - 0.75 + 0.75*(3/2.5) = 1.15
        #       0.1823216 * 1*2.5 / (1 + 1.5*1.15) = 0.1672675
        scores = index.scores("a")
        assert scores[0] == pytest.approx(0.2003533, rel=1e-5)
        assert scores[1] == pytest.approx(0.1672675, rel=1e-5)

    def test_shorter_document_wins_on_an_equal_term(self, index):
        """Length normalisation: one hit in a short passage is stronger evidence
        than one hit in a long one."""
        scores = index.scores("a")
        assert scores[0] > scores[1]

    def test_a_term_in_one_document_scores_only_that_one(self, index):
        scores = index.scores("c")
        assert scores[0] == 0.0
        assert scores[1] == pytest.approx(math.log(2.0) * 2.5 / 2.725, rel=1e-5)

    def test_unseen_terms_contribute_nothing(self, index):
        assert list(index.scores("zzzz")) == [0.0, 0.0]

    def test_empty_corpus_does_not_divide_by_zero(self):
        assert list(BM25Index().fit([]).scores("anything")) == []


class TestSportCheck:
    def test_a_rugby_league_club_is_rejected(self):
        """The regression. English rugby league clubs are named "Football Club",
        so the positive test alone admitted one."""
        rugby = ("Hull Football Club, commonly referred to as Hull F.C., is a "
                 "professional rugby league club based in Kingston upon Hull.")
        assert not is_association_football_club(rugby)

    def test_an_association_football_club_is_accepted(self):
        football = ("Hull City Association Football Club is a professional "
                    "association football club based in Kingston upon Hull.")
        assert is_association_football_club(football)

    def test_a_club_described_without_the_word_association(self):
        """Not every article says "association" — Ipswich Town's does not."""
        assert is_association_football_club(
            "Ipswich Town Football Club is a professional football club based in Ipswich."
        )

    def test_an_unrelated_article_is_rejected(self):
        assert not is_association_football_club(
            "Hull is a port city in the East Riding of Yorkshire, England."
        )


class TestChunking:
    # Every section is comfortably over MIN_WORDS, so a section that gets
    # dropped is always the heading rule working, never the length filter.
    ARTICLE = (
        "Intro paragraph about the club which is written out at some length "
        "here so that it comfortably clears the minimum word filter and is "
        "kept as a chunk in its own right by the chunker under test today.\n"
        "\n== History ==\n"
        "The club was founded in 1904 and this sentence exists purely so that "
        "the section clears the minimum word count that the chunker applies "
        "to every passage before it is stored in the database at all.\n"
        "\n== References ==\n"
        "1. Some citation that nobody ever wants returned as context, padded "
        "out here so that length is never the reason it gets dropped, leaving "
        "the heading rule as the only thing that can remove it from output.\n"
    )

    def test_splits_on_headings(self):
        chunks = chunk_text(self.ARTICLE)
        assert [h for h, _ in chunks] == [None, "History"]

    def test_reference_sections_are_dropped(self):
        """Retrieving a citation list spends the context window on nothing."""
        assert all(h != "References" for h, _ in chunk_text(self.ARTICLE))

    def test_short_fragments_are_dropped(self):
        assert chunk_text("\n== Stub ==\ntoo short\n") == []

    def test_long_sections_are_split(self):
        body = " ".join(f"word{i}" for i in range(MAX_WORDS * 3))
        chunks = chunk_text(f"\n== Long ==\n{body}\n")
        assert len(chunks) > 1
        assert all(len(t.split()) >= MIN_WORDS for _, t in chunks)


@pytest.fixture
def engine(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path / 'rag.db'}", future=True)
    init_db(eng)
    return eng


def _page(title, body):
    return {"title": title, "text": body, "url": f"https://en.wikipedia.org/wiki/{title}"}


# Both are over MIN_WORDS on purpose; shorter fixtures are silently discarded
# by the chunker and every storage assertion then fails for the wrong reason.
LONG_A = ("Arsenal play at the Emirates Stadium which has a capacity of about "
          "sixty thousand supporters and which opened in two thousand and six "
          "after the club moved from Highbury, their home for many decades.")
LONG_B = ("Chelsea play at Stamford Bridge in west London and the ground has "
          "been their home since eighteen ninety five without interruption, "
          "making it one of the oldest continuously used grounds in England.")


class TestStorage:
    def test_document_and_chunks_are_written(self, engine):
        n = store_document(_page("Arsenal F.C.", LONG_A), engine, team="Arsenal")
        assert n == 1
        with engine.connect() as c:
            assert c.execute(text("SELECT COUNT(*) FROM documents")).scalar_one() == 1
            assert c.execute(text("SELECT team FROM documents")).scalar_one() == "Arsenal"
            assert c.execute(text("SELECT license FROM documents")).scalar_one() == "CC BY-SA 4.0"

    def test_reingest_replaces_chunks_rather_than_appending(self, engine):
        """An edited article can lose a section. Merging would keep serving text
        the source no longer has."""
        store_document(_page("Arsenal F.C.", LONG_A), engine, team="Arsenal")
        store_document(_page("Arsenal F.C.", LONG_B), engine, team="Arsenal")
        with engine.connect() as c:
            assert c.execute(text("SELECT COUNT(*) FROM documents")).scalar_one() == 1
            rows = c.execute(text("SELECT text FROM chunks")).all()
        assert len(rows) == 1
        assert "Stamford Bridge" in rows[0][0]


class TestSearch:
    @pytest.fixture
    def corpus(self, engine):
        store_document(_page("Arsenal F.C.", LONG_A), engine, team="Arsenal")
        store_document(_page("Chelsea F.C.", LONG_B), engine, team="Chelsea")
        return engine

    def test_finds_the_relevant_passage(self, corpus):
        hits = search("Emirates Stadium capacity", engine=corpus)
        assert hits and hits[0]["title"] == "Arsenal F.C."

    def test_metadata_filter_restricts_the_pool(self, corpus):
        """Filtering before ranking is the first stage of the cascade. Scoped to
        the wrong club, a query returns fluent, well-ranked, wrong text."""
        # "play" appears in both documents, so an unfiltered search would rank
        # them together — which is what makes this a test of the filter.
        assert len(search("play", engine=corpus)) == 2
        hits = search("play", engine=corpus, team="Chelsea")
        assert hits
        assert all(h["team"] == "Chelsea" for h in hits)

    def test_a_query_matching_nothing_returns_nothing(self, corpus):
        """Padding out to k would invent relevance that is not there."""
        assert search("xylophone quantum tarragon", engine=corpus) == []

    def test_results_are_ranked_and_capped(self, corpus):
        hits = search("stadium", engine=corpus, limit=1)
        assert len(hits) == 1
        assert hits[0]["rank"] == 1

    def test_empty_corpus_is_survivable(self, engine):
        assert search("anything", engine=engine) == []
