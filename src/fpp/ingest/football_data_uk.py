"""Ingest match results and closing odds from football-data.co.uk.

Free CSVs, no API key, no scraping. One file per division per season:
    https://www.football-data.co.uk/mmz4281/{season}/{division}.csv

Two things about this source that cause silent bugs, both handled below:
  * Dates are dd/mm/yyyy. pandas will parse them as US dates without
    dayfirst=True, turning 05/08/2024 into 8 May.
  * B365H/D/A are OPENING odds; the closing line carries a C (B365CH).
    We want closing odds — they are the informed benchmark.
"""

from __future__ import annotations

import io
import logging

import pandas as pd
import requests
from sqlalchemy import Engine, text

from fpp.config import BASE_URL, RAW_DIR
from fpp.db import get_engine
from fpp.ingest.teams import canonical_team_name

log = logging.getLogger(__name__)

REQUIRED = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]

# Preference order for closing odds: market average first, then Bet365.
ODDS_SETS = [
    ("avg_closing", ["AvgCH", "AvgCD", "AvgCA"]),
    ("b365_closing", ["B365CH", "B365CD", "B365CA"]),
]


def fetch_csv(division: str, season: str, use_cache: bool = True) -> pd.DataFrame | None:
    """Download one division-season CSV, caching to disk.

    Fetch once, reuse forever — this keeps request volume respectful and means
    the source being down doesn't block local work.
    """
    cache = RAW_DIR / f"{season}_{division}.csv"
    if use_cache and cache.exists():
        raw = cache.read_bytes()
    else:
        url = f"{BASE_URL}/{season}/{division}.csv"
        log.info("fetching %s", url)
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as exc:
            log.warning("could not fetch %s: %s", url, exc)
            return None
        raw = resp.content
        cache.write_bytes(raw)

    # These files are usually cp1252 and often have trailing blank columns.
    df = pd.read_csv(io.BytesIO(raw), encoding="latin-1", on_bad_lines="skip")
    df = df.dropna(axis=1, how="all")

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        log.warning("%s %s missing columns %s — skipping", division, season, missing)
        return None

    df = df.dropna(subset=REQUIRED)
    # dayfirst=True is load-bearing. See module docstring.
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"])
    df["division"] = division
    df["season"] = season
    return df


def _team_ids(conn, names: set[str], division: str) -> dict[str, int]:
    """Insert any unseen teams, return name -> id for all requested."""
    for name in sorted(names):
        conn.execute(
            text(
                "INSERT INTO teams (name, division) VALUES (:n, :d) "
                "ON CONFLICT (name) DO UPDATE SET division = excluded.division"
            ),
            {"n": name, "d": division},
        )
    rows = conn.execute(text("SELECT name, id FROM teams")).fetchall()
    return {name: tid for name, tid in rows}


def _pick_odds(row: pd.Series) -> tuple[str, float, float, float] | None:
    for source, cols in ODDS_SETS:
        if all(c in row.index and pd.notna(row[c]) for c in cols):
            h, d, a = (float(row[c]) for c in cols)
            if h > 1 and d > 1 and a > 1:
                return source, h, d, a
    return None


def ingest_frame(df: pd.DataFrame, engine: Engine | None = None) -> int:
    """Write one division-season frame to the DB. Idempotent."""
    engine = engine or get_engine()
    division = df["division"].iloc[0]

    df = df.copy()
    df["home_canon"] = df["HomeTeam"].map(canonical_team_name)
    df["away_canon"] = df["AwayTeam"].map(canonical_team_name)
    names = set(df["home_canon"]) | set(df["away_canon"])

    inserted = 0
    with engine.begin() as conn:
        ids = _team_ids(conn, names, division)

        for _, row in df.iterrows():
            params = {
                "division": division,
                "season": row["season"],
                "match_date": row["Date"].date(),
                "home_team_id": ids[row["home_canon"]],
                "away_team_id": ids[row["away_canon"]],
                "home_goals": int(row["FTHG"]),
                "away_goals": int(row["FTAG"]),
                "result": str(row["FTR"]).strip().upper()[:1],
            }
            conn.execute(
                text(
                    """
                    INSERT INTO matches (division, season, match_date, home_team_id,
                                         away_team_id, home_goals, away_goals, result)
                    VALUES (:division, :season, :match_date, :home_team_id,
                            :away_team_id, :home_goals, :away_goals, :result)
                    ON CONFLICT (division, season, match_date, home_team_id, away_team_id)
                    DO UPDATE SET home_goals = excluded.home_goals,
                                  away_goals = excluded.away_goals,
                                  result     = excluded.result
                    """
                ),
                params,
            )
            match_id = conn.execute(
                text(
                    """
                    SELECT id FROM matches
                    WHERE division = :division AND season = :season
                      AND match_date = :match_date
                      AND home_team_id = :home_team_id
                      AND away_team_id = :away_team_id
                    """
                ),
                params,
            ).scalar_one()

            picked = _pick_odds(row)
            if picked:
                source, oh, od, oa = picked
                conn.execute(
                    text(
                        """
                        INSERT INTO odds (match_id, source, odds_home, odds_draw, odds_away)
                        VALUES (:m, :s, :h, :d, :a)
                        ON CONFLICT (match_id) DO UPDATE SET
                            source = excluded.source,
                            odds_home = excluded.odds_home,
                            odds_draw = excluded.odds_draw,
                            odds_away = excluded.odds_away
                        """
                    ),
                    {"m": match_id, "s": source, "h": oh, "d": od, "a": oa},
                )
            inserted += 1
    return inserted


def ingest(
    divisions: list[str],
    seasons: list[str],
    engine: Engine | None = None,
    use_cache: bool = True,
) -> dict[str, int]:
    """Ingest every division-season combination. Missing files are skipped."""
    engine = engine or get_engine()
    counts: dict[str, int] = {}
    for division in divisions:
        for season in seasons:
            df = fetch_csv(division, season, use_cache=use_cache)
            if df is None or df.empty:
                continue
            n = ingest_frame(df, engine)
            counts[f"{division}_{season}"] = n
            log.info("ingested %s rows for %s %s", n, division, season)
    return counts
