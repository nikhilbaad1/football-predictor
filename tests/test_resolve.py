"""Entity-resolution tests.

The one that matters most is TestRefusesToGuess. Every other failure here is
recoverable; a wrong match merges two clubs' histories into one row and the
model then reports a confident number about a team that does not exist.

The real case these are built from: FPL publishes "Hull City" and "Ipswich
Town" where this database has "Hull" and "Ipswich", and also publishes
"Coventry City", a club with no history here at all. Any nearest-match rule
loose enough to fix the first two will match Coventry City to Leicester City.
"""

from __future__ import annotations

import pytest

from fpp.ingest.resolve import CLUB_SUFFIXES, _head, resolve_all, resolve_team_name

KNOWN = {
    "Arsenal", "Chelsea", "Tottenham", "Manchester United", "Manchester City",
    "Leicester City", "Norwich City", "Hull", "Ipswich", "Wolverhampton Wanderers",
}


class TestConfidentMatches:
    def test_exact_name(self):
        r = resolve_team_name("Arsenal", KNOWN)
        assert (r.resolved_to, r.method, r.confidence) == ("Arsenal", "exact", 1.0)

    def test_whitespace_is_normalised(self):
        assert resolve_team_name("  Arsenal  ", KNOWN).resolved_to == "Arsenal"

    @pytest.mark.parametrize(
        "raw,expected",
        [("Spurs", "Tottenham"), ("Man Utd", "Manchester United"),
         ("Man City", "Manchester City"), ("Wolves", "Wolverhampton Wanderers")],
    )
    def test_alias_table(self, raw, expected):
        r = resolve_team_name(raw, KNOWN)
        assert r.resolved_to == expected
        assert r.method == "alias"

    @pytest.mark.parametrize(
        "raw,expected", [("Hull City", "Hull"), ("Ipswich Town", "Ipswich")]
    )
    def test_club_type_suffix_is_not_identity(self, raw, expected):
        """"Hull City" and "Hull" are one club. The suffix says what kind of
        club it is, not which one."""
        r = resolve_team_name(raw, KNOWN)
        assert r.resolved_to == expected
        assert r.method == "head_word"
        assert r.is_resolved


class TestRefusesToGuess:
    def test_a_genuinely_new_club_is_not_matched_to_a_similar_one(self):
        """The case this module exists for.

        "Coventry City" shares a suffix with three known clubs and is a short
        edit from "Leicester City". It is none of them. Resolving it would
        merge two clubs; refusing leaves a question someone can answer.
        """
        r = resolve_team_name("Coventry City", KNOWN)
        assert r.resolved_to is None
        assert not r.is_resolved
        assert r.method == "unresolved"

    def test_the_wrong_answer_is_offered_only_as_a_candidate(self):
        """Near matches are still worth surfacing — as suggestions to a
        decision-maker, never as a decision."""
        r = resolve_team_name("Coventry City", KNOWN)
        assert r.candidates, "a human or agent needs somewhere to start"
        assert r.resolved_to is None

    def test_an_ambiguous_head_word_is_refused(self):
        """Two known clubs sharing an identifying word makes any choice a coin
        flip, so make none."""
        known = KNOWN | {"Manchester"}
        r = resolve_team_name("Manchester Rovers", known)
        assert r.resolved_to is None
        assert "arbitrary" in r.rationale

    def test_empty_name_resolves_to_nothing(self):
        assert resolve_team_name("   ", KNOWN).resolved_to is None

    def test_never_decides_something_is_new(self):
        """The resolver cannot conclude "new club" — that needs knowing which
        clubs exist. It reports absence of a match and stops."""
        for name in ["Coventry City", "Something Entirely Made Up", ""]:
            assert resolve_team_name(name, KNOWN).decision in (None, "matched")


class TestHeadWord:
    @pytest.mark.parametrize(
        "name,head",
        [("Hull City", "hull"), ("Ipswich Town", "ipswich"),
         ("Coventry City", "coventry"), ("Leicester City", "leicester"),
         ("Manchester United", "manchester")],
    )
    def test_strips_club_type_words(self, name, head):
        assert _head(name) == head

    def test_coventry_and_leicester_differ_where_it_counts(self):
        """Both end in "City"; the heads are what tell them apart, which is why
        matching on the head is safe and matching on the whole string is not."""
        assert _head("Coventry City") != _head("Leicester City")

    def test_a_name_that_is_all_suffix_survives(self):
        """Never reduce a name to nothing — that would make every such name
        collide with every other."""
        assert _head("City") == "city"

    def test_suffix_list_is_lowercase(self):
        assert all(s == s.lower() for s in CLUB_SUFFIXES)


class TestResolveAll:
    def test_returns_one_result_per_input(self):
        out = resolve_all(["Arsenal", "Coventry City"], KNOWN)
        assert set(out) == {"Arsenal", "Coventry City"}
        assert out["Arsenal"].is_resolved
        assert not out["Coventry City"].is_resolved
