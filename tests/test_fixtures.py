"""Fixture-ingest tests.

The one that matters most is TestNeverOverwritesAResult. The fixtures feed and
the results files share a uniqueness key, so an ingest written with the wrong
conflict clause would quietly replace finished matches with NULLs — destroying
exactly the data the backtest depends on, without raising anything.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest
from sqlalchemy import text

from fpp.config import season_code
from fpp.db import get_engine, init_db, store_predictions
from fpp.ingest.fixtures import (
    ingest_fixtures,
    parse_fixtures,
    prune_stale_fixtures,
    stale_fixtures,
    teams_without_history,
)

TOMORROW = date.today() + timedelta(days=1)
YESTERDAY = date.today() - timedelta(days=1)


def _csv(rows: list[tuple[str, date, str, str]], bom: bool = True) -> bytes:
    """Build a fixtures CSV shaped like the real one, BOM included by default."""
    header = "Div,Date,Time,HomeTeam,AwayTeam,B365H,B365D,B365A"
    body = "\n".join(
        f"{div},{d.strftime('%d/%m/%Y')},20:00,{home},{away},2.0,3.4,3.9"
        for div, d, home, away in rows
    )
    text_ = f"{header}\n{body}\n"
    return ("﻿" + text_).encode("utf-8") if bom else text_.encode("utf-8")


@pytest.fixture
def engine(tmp_path):
    eng = get_engine(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(eng)
    return eng


def _insert_played(engine, division, match_date, home, away, hg, ag, result):
    with engine.begin() as conn:
        ids = {}
        for name in (home, away):
            conn.execute(
                text("INSERT INTO teams (name, division) VALUES (:n, :d) "
                     "ON CONFLICT (name) DO NOTHING"),
                {"n": name, "d": division},
            )
            ids[name] = conn.execute(
                text("SELECT id FROM teams WHERE name = :n"), {"n": name}
            ).scalar_one()
        conn.execute(
            text(
                """
                INSERT INTO matches (division, season, match_date, home_team_id,
                                     away_team_id, home_goals, away_goals, result)
                VALUES (:div, :season, :d, :h, :a, :hg, :ag, :r)
                """
            ),
            {
                "div": division, "season": season_code(match_date), "d": match_date,
                "h": ids[home], "a": ids[away], "hg": hg, "ag": ag, "r": result,
            },
        )


class TestSeasonCode:
    @pytest.mark.parametrize(
        "d,expected",
        [
            (date(2026, 8, 20), "2627"),   # August is the new season
            (date(2027, 1, 10), "2627"),   # January still belongs to it
            (date(2026, 5, 24), "2526"),   # May is the old one
            (date(2026, 6, 30), "2526"),   # the season boundary is 1 July
            (date(2026, 7, 1), "2627"),
            (date(1999, 9, 1), "9900"),    # century rollover, as the source names it
        ],
    )
    def test_matches_source_naming(self, d, expected):
        assert season_code(d) == expected

    def test_fixture_and_its_result_agree(self):
        """A fixture and the result that follows must land on one season code.

        They form part of the same uniqueness key, so disagreeing here would
        leave a permanently unplayed fixture beside its own finished match.
        """
        assert season_code(date(2026, 8, 20)) == season_code(date(2027, 5, 20))


class TestParsing:
    def test_reads_through_the_byte_order_mark(self):
        """The real file is UTF-8 with a BOM; the season files are not.

        Read as latin-1 the first column is not 'Div', the division filter
        matches nothing, and the ingest silently does nothing at all.
        """
        df = parse_fixtures(_csv([("E0", TOMORROW, "Arsenal", "Chelsea")], bom=True))
        assert list(df.columns)[0] == "Div"
        assert df["Div"].tolist() == ["E0"]

    def test_parses_dates_day_first(self):
        df = parse_fixtures(_csv([("E0", date(2026, 8, 5), "Arsenal", "Chelsea")]))
        parsed = df["Date"].iloc[0]
        assert (parsed.day, parsed.month) == (5, 8)

    def test_missing_columns_returns_none(self):
        assert parse_fixtures(b"Div,Date\nE0,20/08/2026\n") is None


class TestNeverOverwritesAResult:
    def test_played_match_survives_a_colliding_fixture(self, engine):
        """The invariant. A fixture sharing a played match's key must not touch it.

        Same division, season, date and teams — so the fixtures insert conflicts
        on the uniqueness key. With DO UPDATE it would write NULL goals over a
        real 3-1, and every metric computed afterwards would be wrong.
        """
        played_on = date.today() + timedelta(days=2)
        _insert_played(engine, "E0", played_on, "Arsenal", "Chelsea", 3, 1, "H")

        ingest_fixtures(
            ["E0"], engine, df=parse_fixtures(_csv([("E0", played_on, "Arsenal", "Chelsea")]))
        )

        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT home_goals, away_goals, result FROM matches")
            ).fetchone()
            count = conn.execute(text("SELECT COUNT(*) FROM matches")).scalar_one()

        assert row == (3, 1, "H"), "a fixture overwrote a played result"
        assert count == 1, "the fixture was inserted as a duplicate row"

    def test_new_fixture_is_inserted_unplayed(self, engine):
        ingest_fixtures(
            ["E0"], engine, df=parse_fixtures(_csv([("E0", TOMORROW, "Arsenal", "Chelsea")]))
        )
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT result, home_goals FROM matches")
            ).fetchone()
        assert row == (None, None)

    def test_ingest_is_idempotent(self, engine):
        payload = parse_fixtures(_csv([("E0", TOMORROW, "Arsenal", "Chelsea")]))
        ingest_fixtures(["E0"], engine, df=payload)
        ingest_fixtures(["E0"], engine, df=payload)
        with engine.connect() as conn:
            assert conn.execute(text("SELECT COUNT(*) FROM matches")).scalar_one() == 1


class TestFiltering:
    def test_other_divisions_are_ignored(self, engine):
        counts = ingest_fixtures(
            ["E0"], engine,
            df=parse_fixtures(_csv([
                ("E0", TOMORROW, "Arsenal", "Chelsea"),
                ("E2", TOMORROW, "Bradford City", "Sheffield Wednesday"),
            ])),
        )
        assert counts.get("E0") == 1
        assert "E2" not in counts

    def test_past_fixtures_are_not_inserted(self, engine):
        counts = ingest_fixtures(
            ["E0"], engine,
            df=parse_fixtures(_csv([("E0", YESTERDAY, "Arsenal", "Chelsea")])),
        )
        assert counts.get("skipped_past") == 1
        with engine.connect() as conn:
            assert conn.execute(text("SELECT COUNT(*) FROM matches")).scalar_one() == 0

    def test_no_odds_are_stored(self, engine):
        """The feed carries opening prices only. Storing those would put a
        weaker number where the benchmark expects a closing line."""
        ingest_fixtures(
            ["E0"], engine, df=parse_fixtures(_csv([("E0", TOMORROW, "Arsenal", "Chelsea")]))
        )
        with engine.connect() as conn:
            assert conn.execute(text("SELECT COUNT(*) FROM odds")).scalar_one() == 0


class TestStalePostponements:
    def _stale_row(self, engine):
        _insert_played(engine, "E0", YESTERDAY, "Arsenal", "Chelsea", None, None, None)

    def test_counts_unplayed_matches_in_the_past(self, engine):
        self._stale_row(engine)
        assert stale_fixtures(engine) == 1

    def test_prune_removes_them(self, engine):
        self._stale_row(engine)
        assert prune_stale_fixtures(engine) == 1
        assert stale_fixtures(engine) == 0

    def test_prune_leaves_played_matches_alone(self, engine):
        _insert_played(engine, "E0", YESTERDAY, "Arsenal", "Chelsea", 2, 0, "H")
        assert prune_stale_fixtures(engine) == 0
        with engine.connect() as conn:
            assert conn.execute(text("SELECT COUNT(*) FROM matches")).scalar_one() == 1


class TestUnresolvedAliases:
    def test_a_team_with_no_history_is_reported(self, engine):
        """'Atl. Madrid' in the feed against 'Ath Madrid' in the results files
        split one club in two and produced a confident, meaningless prediction.
        Ingest must not raise on it — a promoted club looks identical — so the
        gap has to surface some other way."""
        _insert_played(engine, "SP1", YESTERDAY, "Atletico Madrid", "Malaga", 2, 0, "H")
        ingest_fixtures(
            ["SP1"], engine,
            df=parse_fixtures(_csv([("SP1", TOMORROW, "Atl. Nonexistent", "Malaga")])),
        )
        assert teams_without_history(engine) == ["Atl. Nonexistent"]

    def test_teams_with_history_are_not_reported(self, engine):
        _insert_played(engine, "SP1", YESTERDAY, "Atletico Madrid", "Malaga", 2, 0, "H")
        ingest_fixtures(
            ["SP1"], engine,
            df=parse_fixtures(_csv([("SP1", TOMORROW, "Malaga", "Atletico Madrid")])),
        )
        assert teams_without_history(engine) == []


class TestPredictionsAreLockedAtKickoff:
    """A forward prediction is only evidence if it cannot be rewritten later."""

    def _frame(self, match_id, match_date):
        return pd.DataFrame([{
            "match_id": match_id, "match_date": match_date,
            "model_version": "test-v1",
            "prob_home": 0.5, "prob_draw": 0.3, "prob_away": 0.2,
        }])

    def test_future_fixtures_are_stored(self, engine):
        _insert_played(engine, "E0", TOMORROW, "Arsenal", "Chelsea", None, None, None)
        with engine.connect() as conn:
            mid = conn.execute(text("SELECT id FROM matches")).scalar_one()
        assert store_predictions(self._frame(mid, TOMORROW), engine) == 1

    def test_past_fixtures_are_not_stored(self, engine):
        """Re-running after kick-off must not rewrite what was forecast before it.

        Otherwise every stored prediction silently becomes a backtest with extra
        steps, and the forward record proves nothing.
        """
        _insert_played(engine, "E0", YESTERDAY, "Arsenal", "Chelsea", 1, 1, "D")
        with engine.connect() as conn:
            mid = conn.execute(text("SELECT id FROM matches")).scalar_one()
        assert store_predictions(self._frame(mid, YESTERDAY), engine) == 0
        with engine.connect() as conn:
            assert conn.execute(text("SELECT COUNT(*) FROM predictions")).scalar_one() == 0

    def test_a_stored_prediction_survives_a_later_run(self, engine):
        """The prediction written before kick-off is the one that stays."""
        _insert_played(engine, "E0", TOMORROW, "Arsenal", "Chelsea", None, None, None)
        with engine.connect() as conn:
            mid = conn.execute(text("SELECT id FROM matches")).scalar_one()
        store_predictions(self._frame(mid, TOMORROW), engine)

        # The same match, re-predicted after it has been played.
        late = self._frame(mid, YESTERDAY)
        late.loc[0, "prob_home"] = 0.99
        store_predictions(late, engine)

        with engine.connect() as conn:
            prob = conn.execute(text("SELECT prob_home FROM predictions")).scalar_one()
        assert prob == 0.5
