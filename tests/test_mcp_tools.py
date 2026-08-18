"""MCP tool tests.

These run against a small temporary database rather than the developer's, so
they work in CI where data/football.db does not exist.

The one that matters most is TestReadOnlyIsEnforcedBelowTheFilter. The keyword
filter in run_sql is a courtesy that produces a readable error; the actual
guarantee is that the connection cannot write. If only the filter were tested,
a regex gap would look like a passing suite.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

from fpp.mcp_server import tools
from fpp.models.dixon_coles import DixonColesModel
from fpp.models.elo import EloModel

TOMORROW = date.today() + timedelta(days=1)


@pytest.fixture
def db(tmp_path, monkeypatch, synthetic_matches):
    """A throwaway database wired into the tools, with models fitted on it."""
    path = (tmp_path / "t.db").as_posix()
    rw = create_engine(f"sqlite:///{path}", future=True)

    from fpp.db import init_db
    init_db(rw)

    teams = ["Arsenal", "Chelsea", "Liverpool", "Newly Promoted FC"]
    with rw.begin() as c:
        for n in teams:
            c.execute(text("INSERT INTO teams (name, division) VALUES (:n, 'E0')"), {"n": n})
        ids = {n: c.execute(text("SELECT id FROM teams WHERE name=:n"), {"n": n}).scalar_one()
               for n in teams}

        played = [
            ("2025-08-10", "Arsenal", "Chelsea", 2, 1, "H"),
            ("2025-09-10", "Chelsea", "Arsenal", 0, 0, "D"),
            ("2025-10-10", "Arsenal", "Liverpool", 1, 3, "A"),
            ("2025-11-10", "Liverpool", "Chelsea", 2, 0, "H"),
        ]
        for d, h, a, hg, ag, r in played:
            c.execute(
                text("""INSERT INTO matches (division, season, match_date, home_team_id,
                        away_team_id, home_goals, away_goals, result)
                        VALUES ('E0','2526',:d,:h,:a,:hg,:ag,:r)"""),
                {"d": d, "h": ids[h], "a": ids[a], "hg": hg, "ag": ag, "r": r},
            )
        # one with closing odds, to exercise the de-vig path
        mid = c.execute(text("SELECT id FROM matches ORDER BY id LIMIT 1")).scalar_one()
        c.execute(
            text("""INSERT INTO odds (match_id, source, odds_home, odds_draw, odds_away)
                    VALUES (:m,'avg_closing',2.0,3.5,4.0)"""), {"m": mid},
        )
        # one scheduled fixture, with a locked prediction
        c.execute(
            text("""INSERT INTO matches (division, season, match_date, home_team_id, away_team_id)
                    VALUES ('E0','2627',:d,:h,:a)"""),
            {"d": TOMORROW, "h": ids["Arsenal"], "a": ids["Liverpool"]},
        )
        fid = c.execute(text("SELECT id FROM matches WHERE result IS NULL")).scalar_one()
        c.execute(
            text("""INSERT INTO predictions (match_id, model_version, prob_home, prob_draw,
                    prob_away, created_at) VALUES (:m,'test-v1',0.5,0.3,0.2,CURRENT_TIMESTAMP)"""),
            {"m": fid},
        )

    ro = create_engine(f"sqlite:///file:{path}?mode=ro&uri=true", future=True)
    monkeypatch.setattr(tools, "read_only_engine", lambda: ro)
    tools._team_names.cache_clear()

    # Fit on the synthetic fixture: fast, and the tools only need *a* model.
    elo = EloModel().fit(synthetic_matches)
    dc = DixonColesModel().fit(synthetic_matches)
    monkeypatch.setattr(tools, "_models", lambda: (elo, dc))

    yield ro
    tools._team_names.cache_clear()


class TestTeamResolution:
    def test_resolves_a_known_alias(self, db):
        assert tools.resolve_team("Arsenal") == "Arsenal"

    def test_is_case_insensitive(self, db):
        assert tools.resolve_team("arsenal") == "Arsenal"

    def test_unknown_name_suggests_the_closest(self, db):
        """The error has to be recoverable. Canonical spellings are not
        guessable, so a bare failure just makes the caller guess again."""
        with pytest.raises(tools.ToolError) as exc:
            tools.resolve_team("Arsenl")
        assert "Arsenal" in str(exc.value)

    def test_empty_name_is_rejected(self, db):
        with pytest.raises(tools.ToolError):
            tools.resolve_team("   ")


class TestNamedTools:
    def test_head_to_head_is_order_independent(self, db):
        a = tools.get_head_to_head("Arsenal", "Chelsea")
        b = tools.get_head_to_head("Chelsea", "Arsenal")
        assert a["meetings_returned"] == b["meetings_returned"] == 2

    def test_head_to_head_counts_the_record(self, db):
        r = tools.get_head_to_head("Arsenal", "Chelsea")["record"]
        assert r == {"Arsenal wins": 1, "draws": 1, "Chelsea wins": 0}

    def test_head_to_head_devigs_the_market_price(self, db):
        """Odds are stored as decimal prices; the caller wants probabilities,
        and wants them with the bookmaker's margin already removed.

        The raw prices imply 1/2.0 + 1/3.5 + 1/4.0 = 1.036, so the margin is
        real and has to go. Tolerance is 1e-3 because the output is rounded to
        four places for readability, the same convention predict_fixtures uses.
        """
        matches = tools.get_head_to_head("Arsenal", "Chelsea")["matches"]
        priced = [m for m in matches if "market_probs" in m]
        assert len(priced) == 1
        probs = priced[0]["market_probs"]
        assert sum(probs) == pytest.approx(1.0, abs=1e-3)
        assert probs[0] > probs[1] > probs[2]  # 2.0 is the shortest price

    def test_team_form_reads_most_recent_first(self, db):
        form = tools.get_team_form("Arsenal")
        assert form["form"][0] == "L"  # 1-3 to Liverpool, the latest played
        assert form["goals_for"] == 3 and form["goals_against"] == 4

    def test_team_form_explains_a_team_with_no_history(self, db):
        with pytest.raises(tools.ToolError) as exc:
            tools.get_team_form("Newly Promoted FC")
        assert "promoted" in str(exc.value)

    def test_upcoming_returns_the_locked_prediction(self, db):
        """Not a fresh one. The stored row is what was forecast before kick-off,
        and reporting anything else misrepresents the record (ADR 0010)."""
        out = tools.get_upcoming_fixtures("E0")
        assert out["count"] == 1
        assert out["fixtures"][0]["prob_home"] == 0.5
        assert out["fixtures"][0]["model_version"] == "test-v1"

    def test_unknown_division_is_rejected(self, db):
        with pytest.raises(tools.ToolError):
            tools.get_upcoming_fixtures("ZZ9")

    def test_predict_match_returns_three_outcomes_summing_to_one(self, db):
        """Non-negotiable #2: three outcomes, never a single win probability.

        Tolerance is 1e-3 because each probability is rounded to four places on
        the way out, so they sum to 1 only to that precision.
        """
        res = tools.predict_match("Arsenal", "Chelsea")
        assert set(res) >= {"prob_home", "prob_draw", "prob_away"}
        total = res["prob_home"] + res["prob_draw"] + res["prob_away"]
        assert total == pytest.approx(1.0, abs=1e-3)

    def test_predict_match_resolves_aliases(self, db):
        res = tools.predict_match("arsenal", "Chelsea")
        assert res["home_team"] == "Arsenal"

    def test_a_team_cannot_play_itself(self, db):
        with pytest.raises(tools.ToolError):
            tools.predict_match("Arsenal", "Arsenal")


class TestRunSqlGuards:
    def test_select_works(self, db):
        out = tools.run_sql("SELECT COUNT(*) AS n FROM matches")
        assert out["rows"][0]["n"] == 5

    def test_with_clause_is_allowed(self, db):
        assert tools.run_sql("WITH x AS (SELECT 1 AS a) SELECT * FROM x")["row_count"] == 1

    @pytest.mark.parametrize(
        "query",
        [
            "INSERT INTO teams (name) VALUES ('x')",
            "UPDATE teams SET name = 'x' WHERE id = 1",
            "DELETE FROM teams WHERE id = 1",
            "DROP TABLE matches",
            "ALTER TABLE matches ADD COLUMN x INTEGER",
            "CREATE TABLE evil (id INTEGER)",
            "PRAGMA table_info(matches)",
            "ATTACH DATABASE 'other.db' AS o",
        ],
    )
    def test_non_read_statements_are_refused(self, db, query):
        with pytest.raises(tools.ToolError):
            tools.run_sql(query)

    def test_stacked_statements_are_refused(self, db):
        """One statement at a time, so a SELECT cannot smuggle a second verb."""
        with pytest.raises(tools.ToolError):
            tools.run_sql("SELECT 1; DROP TABLE matches")

    def test_empty_query_is_refused(self, db):
        with pytest.raises(tools.ToolError):
            tools.run_sql("   ")

    def test_rows_are_capped_and_flagged(self, db):
        out = tools.run_sql("SELECT id FROM matches", limit=2)
        assert out["row_count"] == 2
        assert out["truncated"] is True

    def test_limit_cannot_exceed_the_ceiling(self, db):
        out = tools.run_sql("SELECT id FROM matches", limit=10_000)
        assert out["row_count"] <= tools.MAX_ROWS

    def test_a_broken_query_returns_a_readable_error(self, db):
        with pytest.raises(tools.ToolError) as exc:
            tools.run_sql("SELECT * FROM no_such_table")
        assert "Query failed" in str(exc.value)


class TestReadOnlyIsEnforcedBelowTheFilter:
    """The keyword filter is a courtesy; the connection is the guarantee.

    The PreToolUse hook from ADR 0009 inspects shell commands and cannot see an
    MCP call, so this layer has to hold on its own. If the guarantee lived only
    in the regex, a gap in it would be a silent write path.
    """

    def test_the_connection_itself_refuses_writes(self, db):
        with pytest.raises(OperationalError, match="readonly|read-only"):
            with db.begin() as conn:
                conn.execute(text("UPDATE teams SET name = 'hacked' WHERE id = 1"))

    def test_data_is_unchanged_after_the_attempt(self, db):
        with pytest.raises(OperationalError, match="readonly|read-only"):
            with db.begin() as conn:
                conn.execute(text("DELETE FROM teams WHERE id = 1"))
        with db.connect() as conn:
            assert conn.execute(text("SELECT COUNT(*) FROM teams")).scalar_one() == 4
