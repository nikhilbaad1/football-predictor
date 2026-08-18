"""The tools themselves. Plain Python, no MCP imports.

Kept separate from server.py so the interesting logic is testable without an
MCP client, and so the protocol layer stays a thin adapter that adds nothing
worth testing.

Two things shape every function here.

**Purposeful tools, not one execute_sql.** A single raw-SQL tool pushes schema
knowledge onto the caller, makes every call a possible table scan, and leaves
nothing to evaluate. The four named tools below answer the questions this
database actually gets asked; run_sql exists for the rest and is deliberately
the least convenient option (ADR 0003, PLAN section 4).

**Read-only is enforced by the connection, not by good intentions.** The
PreToolUse hook from ADR 0009 inspects shell commands; an MCP tool call never
touches a shell, so that guard cannot see it. The engine here is opened with
SQLite's mode=ro URI, which fails writes inside the driver.
"""

from __future__ import annotations

import difflib
import re
import time
from datetime import date
from functools import lru_cache
from typing import Any

import numpy as np
from sqlalchemy import Engine, create_engine, text

from fpp.config import DATA_DIR, DATABASE_URL, DISPLAY_DIVISION, DIVISIONS
from fpp.db import load_matches
from fpp.evaluation import remove_vig
from fpp.ingest.teams import canonical_team_name
from fpp.predict import MODEL_VERSION, Fixture, fit_models, predict_fixtures

MAX_ROWS = 200
SQL_TIMEOUT_SECONDS = 5.0


class ToolError(ValueError):
    """A caller-fixable problem. The message is the tool's response."""


# --------------------------------------------------------------------------
# connections and models
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def read_only_engine() -> Engine:
    """A connection that cannot write, enforced by the driver.

    For SQLite this is the mode=ro URI, which makes any write fail with
    "attempt to write a readonly database" inside sqlite3 itself. Postgres
    would use a role with no write grants; the point is that the guarantee
    lives below this code rather than in it.
    """
    if DATABASE_URL.startswith("sqlite"):
        path = (DATA_DIR / "football.db").as_posix()
        return create_engine(f"sqlite:///file:{path}?mode=ro&uri=true", future=True)
    return create_engine(DATABASE_URL, future=True, execution_options={"postgresql_readonly": True})


@lru_cache(maxsize=1)
def _models():
    """Fit once, reuse. Roughly 2.5s on the current data.

    Lazy rather than at import: the three history tools never need a model, and
    making every session pay 2.5s to start the server so that one tool might be
    faster is the wrong trade.
    """
    played = load_matches(divisions=list(DIVISIONS), played_only=True)
    if played.empty:
        raise ToolError("The database has no matches. Run scripts/ingest.py first.")
    return fit_models(played)


@lru_cache(maxsize=1)
def _team_names() -> tuple[str, ...]:
    with read_only_engine().connect() as conn:
        return tuple(r[0] for r in conn.execute(text("SELECT name FROM teams ORDER BY name")))


def resolve_team(raw: str) -> str:
    """Canonical team name, or an error that says what to try instead.

    Callers guess names, and the canonical spellings are not guessable — this
    database says "Tottenham", not "Tottenham Hotspur". Aliases are tried first,
    then case-insensitive match, then near-misses. Returning the closest names
    in the error is the difference between a tool the caller can recover from
    and one that just fails.
    """
    if not raw or not raw.strip():
        raise ToolError("No team name given.")

    names = _team_names()
    canon = canonical_team_name(raw)
    if canon in names:
        return canon

    lowered = {n.lower(): n for n in names}
    if canon.lower() in lowered:
        return lowered[canon.lower()]

    close = difflib.get_close_matches(canon, names, n=5, cutoff=0.6)
    if not close:
        close = [n for n in names if canon.lower() in n.lower()][:5]
    hint = f" Closest matches: {', '.join(close)}." if close else ""
    raise ToolError(f"No team called {raw!r} in the database.{hint}")


# --------------------------------------------------------------------------
# tools
# --------------------------------------------------------------------------


def predict_match(home: str, away: str) -> dict[str, Any]:
    """Three-way probabilities for any pairing, scheduled or hypothetical."""
    home_c, away_c = resolve_team(home), resolve_team(away)
    if home_c == away_c:
        raise ToolError("A team cannot play itself.")

    elo, dc = _models()
    row = predict_fixtures([Fixture(home_c, away_c)], elo, dc).iloc[0].to_dict()
    row.pop("match_id", None)
    row.pop("match_date", None)
    return {
        **row,
        "elo_home": round(elo.rating(home_c), 1),
        "elo_away": round(elo.rating(away_c), 1),
        "note": "Probabilistic estimate from historical results. Not betting advice.",
    }


def get_upcoming_fixtures(division: str | None = None, limit: int = 20) -> dict[str, Any]:
    """Scheduled fixtures and the predictions locked in for them.

    Returns stored predictions rather than fresh ones. Those rows are the
    record written before kick-off (ADR 0010); recomputing here would report a
    number that was never the forecast.
    """
    division = division or DISPLAY_DIVISION
    if division not in DIVISIONS:
        raise ToolError(f"Unknown division {division!r}. Known: {', '.join(DIVISIONS)}.")

    sql = text(
        """
        SELECT m.match_date, h.name AS home_team, a.name AS away_team,
               p.prob_home, p.prob_draw, p.prob_away, p.model_version, p.created_at
        FROM matches m
        JOIN teams h ON h.id = m.home_team_id
        JOIN teams a ON a.id = m.away_team_id
        LEFT JOIN predictions p ON p.match_id = m.id
        WHERE m.result IS NULL AND m.division = :div AND m.match_date >= :today
        ORDER BY m.match_date, h.name
        LIMIT :lim
        """
    )
    with read_only_engine().connect() as conn:
        rows = conn.execute(
            sql,
            {"div": division, "today": date.today(), "lim": min(limit, MAX_ROWS)},
        ).mappings().all()

    fixtures = [dict(r) for r in rows]
    out = {
        "division": DIVISIONS[division],
        "count": len(fixtures),
        "fixtures": fixtures,
    }
    if not fixtures:
        out["note"] = (
            "Nothing scheduled. The source publishes about a week ahead, so this is "
            "normal between rounds and before a season starts. Refresh with "
            "scripts/fixtures.py."
        )
    elif any(f["prob_home"] is None for f in fixtures):
        out["note"] = (
            "Some fixtures have no stored prediction yet — run scripts/fixtures.py."
        )
    return out


def get_head_to_head(team_a: str, team_b: str, limit: int = 10) -> dict[str, Any]:
    """Past meetings between two clubs, most recent first.

    Includes the de-vigged closing price where the source carried one, so the
    market's view sits next to the result rather than needing a second call.
    """
    a, b = resolve_team(team_a), resolve_team(team_b)
    if a == b:
        raise ToolError("A team cannot play itself.")

    sql = text(
        """
        SELECT m.match_date, m.division, m.season,
               h.name AS home_team, a.name AS away_team,
               m.home_goals, m.away_goals, m.result,
               o.odds_home, o.odds_draw, o.odds_away
        FROM matches m
        JOIN teams h ON h.id = m.home_team_id
        JOIN teams a ON a.id = m.away_team_id
        LEFT JOIN odds o ON o.match_id = m.id
        WHERE m.result IS NOT NULL
          AND ((h.name = :a AND a.name = :b) OR (h.name = :b AND a.name = :a))
        ORDER BY m.match_date DESC
        LIMIT :lim
        """
    )
    with read_only_engine().connect() as conn:
        rows = [dict(r) for r in conn.execute(
            sql, {"a": a, "b": b, "lim": min(limit, MAX_ROWS)}
        ).mappings().all()]

    wins = {a: 0, b: 0, "draw": 0}
    for r in rows:
        if r["result"] == "D":
            wins["draw"] += 1
        else:
            winner = r["home_team"] if r["result"] == "H" else r["away_team"]
            wins[winner] += 1
        odds = (r.pop("odds_home"), r.pop("odds_draw"), r.pop("odds_away"))
        if all(o is not None for o in odds):
            p = remove_vig(np.array([odds], dtype=float))[0]
            r["market_probs"] = [round(float(x), 4) for x in p]

    return {
        "teams": [a, b],
        "meetings_returned": len(rows),
        "record": {f"{a} wins": wins[a], "draws": wins["draw"], f"{b} wins": wins[b]},
        "matches": rows,
        "note": (
            "Head-to-head samples are small and mostly reflect squads that have since "
            "turned over. Treat as context, not as a standalone signal."
        ),
    }


def get_team_form(team: str, n: int = 10) -> dict[str, Any]:
    """Recent results for one club, plus its current Elo rating."""
    name = resolve_team(team)
    n = max(1, min(n, MAX_ROWS))

    sql = text(
        """
        SELECT m.match_date, m.division, h.name AS home_team, a.name AS away_team,
               m.home_goals, m.away_goals, m.result
        FROM matches m
        JOIN teams h ON h.id = m.home_team_id
        JOIN teams a ON a.id = m.away_team_id
        WHERE m.result IS NOT NULL AND :name IN (h.name, a.name)
        ORDER BY m.match_date DESC
        LIMIT :lim
        """
    )
    with read_only_engine().connect() as conn:
        rows = [dict(r) for r in conn.execute(sql, {"name": name, "lim": n}).mappings().all()]

    if not rows:
        raise ToolError(
            f"{name!r} exists but has no played matches. It is either newly "
            f"promoted or an unresolved name alias — see teams_without_history."
        )

    form, scored, conceded = [], 0, 0
    for r in rows:
        at_home = r["home_team"] == name
        gf = r["home_goals"] if at_home else r["away_goals"]
        ga = r["away_goals"] if at_home else r["home_goals"]
        scored, conceded = scored + gf, conceded + ga
        form.append("W" if gf > ga else "D" if gf == ga else "L")

    elo, _ = _models()
    return {
        "team": name,
        "matches_returned": len(rows),
        "form": "".join(form),  # most recent first
        "goals_for": scored,
        "goals_against": conceded,
        "elo": round(elo.rating(name), 1),
        "model_version": MODEL_VERSION,
        "matches": rows,
    }


# --------------------------------------------------------------------------
# the escape hatch
# --------------------------------------------------------------------------

_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|TRUNCATE|ATTACH|DETACH|PRAGMA|VACUUM)\b",
    re.I,
)


def run_sql(query: str, limit: int = 100) -> dict[str, Any]:
    """Read-only SQL for questions the four named tools do not cover.

    Guarded three ways, because each catches what the others miss. The engine is
    read-only, so a write fails inside the driver even if everything above is
    wrong -- that is the guarantee. The keyword and single-statement checks come
    first only to return a clear error instead of a driver exception. The row cap
    and timeout bound the damage a careless query does to the caller's context.

    Schema: teams(id, name, division); matches(id, division, season, match_date,
    home_team_id, away_team_id, home_goals, away_goals, result);
    odds(match_id, source, odds_home, odds_draw, odds_away);
    predictions(id, match_id, model_version, prob_home, prob_draw, prob_away,
    created_at). Goals and result are NULL for fixtures not yet played.
    """
    q = query.strip().rstrip(";").strip()
    if not q:
        raise ToolError("Empty query.")
    if ";" in q:
        raise ToolError("One statement at a time.")
    if not re.match(r"^\s*(SELECT|WITH)\b", q, re.I):
        raise ToolError("Only SELECT (or WITH ... SELECT) queries are allowed.")
    if _FORBIDDEN.search(q):
        raise ToolError("Only read queries are allowed; this database is opened read-only.")

    limit = max(1, min(limit, MAX_ROWS))
    with read_only_engine().connect() as conn:
        # Abort a runaway query rather than hang the caller. sqlite3 checks the
        # progress handler every N virtual-machine instructions; returning
        # non-zero raises OperationalError("interrupted").
        deadline = time.monotonic() + SQL_TIMEOUT_SECONDS
        raw = conn.connection.dbapi_connection
        raw.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
        try:
            rows = conn.execute(text(q)).mappings().fetchmany(limit)
        except Exception as exc:  # noqa: BLE001 - surfaced to the caller as text
            raise ToolError(f"Query failed: {exc}") from exc
        finally:
            raw.set_progress_handler(None, 0)

    out = [dict(r) for r in rows]
    return {
        "row_count": len(out),
        "rows": out,
        "truncated": len(out) == limit,
    }
