"""Database access. Everything that touches the DB goes through here."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, create_engine, text

from fpp.config import DATABASE_URL

log = logging.getLogger(__name__)

_SCHEMA = Path(__file__).parent / "schema.sql"


def get_engine(url: str | None = None) -> Engine:
    return create_engine(url or DATABASE_URL, future=True)


def init_db(engine: Engine | None = None) -> None:
    """Create tables if absent. Safe to run repeatedly."""
    engine = engine or get_engine()
    statements = [s.strip() for s in _SCHEMA.read_text().split(";") if s.strip()]
    with engine.begin() as conn:
        for stmt in statements:
            conn.execute(text(stmt))


def load_matches(
    engine: Engine | None = None,
    divisions: list[str] | None = None,
    played_only: bool = True,
) -> pd.DataFrame:
    """Matches joined to team names and closing odds, ordered by date.

    Returns columns: match_id, division, season, match_date, home_team,
    away_team, home_goals, away_goals, result, odds_home, odds_draw, odds_away.
    """
    engine = engine or get_engine()
    where = []
    params: dict = {}
    if played_only:
        where.append("m.result IS NOT NULL")
    if divisions:
        placeholders = ", ".join(f":div{i}" for i in range(len(divisions)))
        where.append(f"m.division IN ({placeholders})")
        params.update({f"div{i}": d for i, d in enumerate(divisions)})
    clause = f"WHERE {' AND '.join(where)}" if where else ""

    sql = f"""
        SELECT m.id AS match_id, m.division, m.season, m.match_date,
               h.name AS home_team, a.name AS away_team,
               m.home_goals, m.away_goals, m.result,
               o.odds_home, o.odds_draw, o.odds_away
        FROM matches m
        JOIN teams h ON h.id = m.home_team_id
        JOIN teams a ON a.id = m.away_team_id
        LEFT JOIN odds o ON o.match_id = m.id
        {clause}
        ORDER BY m.match_date, m.id
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn, params=params)

    # SQLite has no DATE type and hands back ISO strings; Postgres returns
    # date objects. Normalize so downstream code never has to care (ADR 0005).
    df["match_date"] = pd.to_datetime(df["match_date"]).dt.date
    return df


def store_predictions(
    df: pd.DataFrame,
    engine: Engine | None = None,
    today: date | None = None,
) -> int:
    """Write predictions, but only for matches that have not kicked off.

    This is the rule that makes a forward prediction worth anything. Re-running
    the predictor is normal and should refresh a fixture's numbers as new
    results arrive — but only while the match is still in the future. Once its
    date has passed the stored row is the record of what was forecast without
    knowing the answer, and rewriting it would turn the whole exercise into a
    backtest with extra steps.

    Rows with a past `match_date` are dropped rather than written, and the count
    returned is the number actually stored. Expects match_id, match_date,
    model_version and prob_* columns.
    """
    if df.empty:
        return 0
    today = today or date.today()
    dates = pd.to_datetime(df["match_date"]).dt.date
    future = df[dates >= today]
    if len(future) < len(df):
        log.info(
            "not storing %s prediction(s) for matches already played or under way",
            len(df) - len(future),
        )
    return upsert_predictions(
        future[["match_id", "model_version", "prob_home", "prob_draw", "prob_away"]],
        engine,
    )


def upsert_predictions(df: pd.DataFrame, engine: Engine | None = None) -> int:
    """Insert or replace predictions. Expects match_id, model_version, prob_*."""
    engine = engine or get_engine()
    rows = df.to_dict("records")
    if not rows:
        return 0
    sql = text(
        """
        INSERT INTO predictions
            (match_id, model_version, prob_home, prob_draw, prob_away, created_at)
        VALUES (:match_id, :model_version, :prob_home, :prob_draw, :prob_away,
                CURRENT_TIMESTAMP)
        ON CONFLICT (match_id, model_version) DO UPDATE SET
            prob_home = excluded.prob_home,
            prob_draw = excluded.prob_draw,
            prob_away = excluded.prob_away,
            created_at = CURRENT_TIMESTAMP
        """
    )
    with engine.begin() as conn:
        conn.execute(sql, rows)
    return len(rows)
