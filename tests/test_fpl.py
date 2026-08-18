"""FPL ingest tests.

Two invariants carry real weight here.

TestUnresolvedSquadsAreSkipped: a club whose identity is not settled must not
have its players written, because attaching a squad to the wrong team is
invisible afterwards — the rows all look valid.

TestAvailabilityIsATimeSeries: availability feeds expected minutes, which feeds
prediction. If today's snapshot overwrote yesterday's, a backtest would fit on
knowledge that did not exist at the time, which is non-negotiable #1.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine, text

from fpp.db import init_db
from fpp.ingest.fpl import ingest_fpl, pending_resolutions

TODAY = date.today()
YESTERDAY = TODAY - timedelta(days=1)


def bootstrap(teams, elements):
    """A minimal payload shaped like the real bootstrap-static response."""
    return {
        "teams": teams,
        "elements": elements,
        "element_types": [
            {"id": 1, "singular_name_short": "GKP"},
            {"id": 2, "singular_name_short": "DEF"},
            {"id": 3, "singular_name_short": "MID"},
            {"id": 4, "singular_name_short": "FWD"},
        ],
    }


def player(fpl_id, team, name, status="a", chance=None, news=None, etype=3):
    return {
        "id": fpl_id, "team": team, "element_type": etype,
        "first_name": "A", "second_name": name, "web_name": name,
        "status": status, "chance_of_playing_next_round": chance, "news": news,
    }


@pytest.fixture
def engine(tmp_path):
    eng = create_engine(f"sqlite:///{tmp_path / 'fpl.db'}", future=True)
    init_db(eng)
    with eng.begin() as c:
        for n in ("Arsenal", "Hull"):
            c.execute(text("INSERT INTO teams (name, division) VALUES (:n,'E0')"), {"n": n})
    return eng


class TestPlayerIngest:
    def test_players_land_against_the_right_team(self, engine):
        data = bootstrap(
            [{"id": 1, "name": "Arsenal"}],
            [player(10, 1, "Saka"), player(11, 1, "Rice")],
        )
        out = ingest_fpl(engine, data=data)
        assert out["players"] == 2
        with engine.connect() as c:
            rows = c.execute(
                text("""SELECT p.web_name, t.name FROM players p
                        JOIN teams t ON t.id = p.team_id ORDER BY p.web_name""")
            ).all()
        assert rows == [("Rice", "Arsenal"), ("Saka", "Arsenal")]

    def test_a_suffix_variant_still_finds_its_team(self, engine):
        """FPL says "Hull City"; this database says "Hull". One club."""
        data = bootstrap([{"id": 1, "name": "Hull City"}], [player(10, 1, "Smith")])
        ingest_fpl(engine, data=data)
        with engine.connect() as c:
            assert c.execute(
                text("""SELECT t.name FROM players p JOIN teams t ON t.id=p.team_id""")
            ).scalar_one() == "Hull"

    def test_position_is_mapped_from_element_type(self, engine):
        data = bootstrap([{"id": 1, "name": "Arsenal"}], [player(10, 1, "Raya", etype=1)])
        ingest_fpl(engine, data=data)
        with engine.connect() as c:
            assert c.execute(text("SELECT position FROM players")).scalar_one() == "GKP"

    def test_rerunning_does_not_duplicate(self, engine):
        data = bootstrap([{"id": 1, "name": "Arsenal"}], [player(10, 1, "Saka")])
        ingest_fpl(engine, data=data)
        ingest_fpl(engine, data=data)
        with engine.connect() as c:
            assert c.execute(text("SELECT COUNT(*) FROM players")).scalar_one() == 1

    def test_a_transfer_moves_the_player(self, engine):
        ingest_fpl(engine, data=bootstrap(
            [{"id": 1, "name": "Arsenal"}, {"id": 2, "name": "Hull"}],
            [player(10, 1, "Smith")]))
        ingest_fpl(engine, data=bootstrap(
            [{"id": 1, "name": "Arsenal"}, {"id": 2, "name": "Hull"}],
            [player(10, 2, "Smith")]))
        with engine.connect() as c:
            assert c.execute(
                text("SELECT t.name FROM players p JOIN teams t ON t.id=p.team_id")
            ).scalar_one() == "Hull"


class TestUnresolvedSquadsAreSkipped:
    def test_players_of_an_unresolved_club_are_not_written(self, engine):
        """Better to be missing a squad than to attach it to the wrong club.

        A misattached squad produces rows that all look valid, so nothing
        downstream ever surfaces the mistake.
        """
        data = bootstrap(
            [{"id": 1, "name": "Arsenal"}, {"id": 2, "name": "Coventry City"}],
            [player(10, 1, "Saka"), player(20, 2, "Unknown")],
        )
        out = ingest_fpl(engine, data=data)

        assert out["unresolved"] == ["Coventry City"]
        assert out["players"] == 1
        with engine.connect() as c:
            assert c.execute(text("SELECT web_name FROM players")).scalar_one() == "Saka"

    def test_the_unresolved_name_is_queued_with_its_reasoning(self, engine):
        data = bootstrap([{"id": 2, "name": "Coventry City"}], [])
        ingest_fpl(engine, data=data)
        pending = pending_resolutions(engine)
        assert [p["raw_name"] for p in pending] == ["Coventry City"]
        assert pending[0]["rationale"]

    def test_resolved_names_are_recorded_too(self, engine):
        """Known-good decisions are the eval set the agent gets scored against."""
        ingest_fpl(engine, data=bootstrap([{"id": 1, "name": "Arsenal"}], []))
        with engine.connect() as c:
            row = c.execute(
                text("""SELECT resolved_to, decision, method FROM name_resolutions
                        WHERE raw_name = 'Arsenal'""")
            ).one()
        assert row == ("Arsenal", "matched", "exact")
        assert pending_resolutions(engine) == []


class TestAvailabilityIsATimeSeries:
    def _data(self, status, chance, news=None):
        return bootstrap(
            [{"id": 1, "name": "Arsenal"}],
            [player(10, 1, "Saka", status=status, chance=chance, news=news)],
        )

    def test_each_run_writes_a_dated_snapshot(self, engine):
        ingest_fpl(engine, data=self._data("a", 100), today=YESTERDAY)
        ingest_fpl(engine, data=self._data("i", 0, "Hamstring"), today=TODAY)
        with engine.connect() as c:
            rows = c.execute(
                text("""SELECT as_of, status, chance_next_round
                        FROM player_availability ORDER BY as_of""")
            ).all()
        assert len(rows) == 2, "today's snapshot overwrote yesterday's"

    def test_yesterdays_knowledge_is_still_yesterdays(self, engine):
        """The leakage guard. A model predicting yesterday's match must see the
        player as available, because that is what was known then — even though
        he is injured today.
        """
        ingest_fpl(engine, data=self._data("a", 100), today=YESTERDAY)
        ingest_fpl(engine, data=self._data("i", 0, "Hamstring"), today=TODAY)
        with engine.connect() as c:
            was = c.execute(
                text("SELECT status FROM player_availability WHERE as_of = :d"),
                {"d": YESTERDAY},
            ).scalar_one()
        assert was == "a"

    def test_the_same_day_refreshes_rather_than_duplicating(self, engine):
        ingest_fpl(engine, data=self._data("a", 100), today=TODAY)
        ingest_fpl(engine, data=self._data("d", 50, "Knock"), today=TODAY)
        with engine.connect() as c:
            rows = c.execute(
                text("SELECT status, chance_next_round FROM player_availability")
            ).all()
        assert rows == [("d", 50)]

    def test_missing_chance_is_null_not_zero(self, engine):
        """FPL sends null when it has no view. Storing that as 0 would read as
        "definitely not playing", which is a different claim entirely."""
        ingest_fpl(engine, data=self._data("a", None), today=TODAY)
        with engine.connect() as c:
            assert c.execute(
                text("SELECT chance_next_round FROM player_availability")
            ).scalar_one() is None


class TestEmptyOrBrokenPayloads:
    def test_missing_payload_is_survivable(self, engine):
        assert ingest_fpl(engine, data={})["players"] == 0

    def test_payload_without_elements_is_survivable(self, engine):
        assert ingest_fpl(engine, data={"teams": []})["players"] == 0
