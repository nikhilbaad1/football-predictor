"""Ingest upcoming fixtures from football-data.co.uk.

    https://www.football-data.co.uk/fixtures.csv

One file, every division, roughly the next week. This is what lets the project
predict forward rather than only re-score history, which is the difference
between a backtest and something that can be graded by the season.

Four things about this file differ from the per-season results files, and each
is a silent failure rather than an error if missed:

  * It is UTF-8 with a byte-order mark. Read as latin-1 like the season files
    and the first column name arrives with the BOM glued to the front, so it no
    longer equals "Div", the division filter matches nothing, and the ingest
    quietly does nothing at all.
  * There is no season column. It is derived from the date via
    config.season_code, which has to agree with how the source names its
    results files, or a fixture never reconciles with its own result.
  * There are no closing odds. The C-suffixed columns appear in the header but
    are empty, because a closing line does not exist until shortly before
    kick-off. Only opening prices are populated, and those are deliberately not
    stored -- see CLAUDE.md on the B365H / B365CH trap.
  * Dates are dd/mm/yyyy, the same trap as everywhere else in this source.
"""

from __future__ import annotations

import io
import logging
from datetime import date

import pandas as pd
import requests
from sqlalchemy import Engine, text

from fpp.config import FIXTURES_URL, season_code
from fpp.db import get_engine

# Module-private helper shared inside fpp.ingest: inserts unseen teams and
# returns name -> id. Fixtures reuse it so a club appearing in a fixture before
# its first result gets the same team row rather than a second one.
from fpp.ingest.football_data_uk import _team_ids
from fpp.ingest.teams import canonical_team_name

log = logging.getLogger(__name__)

REQUIRED = ["Div", "Date", "HomeTeam", "AwayTeam"]


def parse_fixtures(content: bytes) -> pd.DataFrame | None:
    """Parse the raw fixtures CSV. Split from fetching so it can be tested."""
    # utf-8-sig, not latin-1. See the module docstring.
    df = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig", on_bad_lines="skip")
    df = df.dropna(axis=1, how="all")

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        log.warning("fixtures file is missing columns %s -- skipping", missing)
        return None

    df = df.dropna(subset=REQUIRED)
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    return df.dropna(subset=["Date"])


def fetch_fixtures(url: str = FIXTURES_URL) -> pd.DataFrame | None:
    """Download the fixtures file. Deliberately never cached.

    Season results files are cached forever because a played match does not
    change. A fixture list does: matches are added, postponed and rescheduled,
    so serving a stale copy would defeat the purpose of having it.
    """
    log.info("fetching %s", url)
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.warning("could not fetch %s: %s", url, exc)
        return None
    return parse_fixtures(resp.content)


def ingest_fixtures(
    divisions: list[str],
    engine: Engine | None = None,
    df: pd.DataFrame | None = None,
    today: date | None = None,
) -> dict[str, int]:
    """Insert unplayed fixtures for the given divisions. Idempotent.

    Returns new-row counts keyed by division, plus "skipped_past" when the feed
    contained fixtures whose date has already gone.
    """
    engine = engine or get_engine()
    today = today or date.today()
    if df is None:
        df = fetch_fixtures()
    if df is None or df.empty:
        return {}

    df = df[df["Div"].isin(divisions)].copy()
    if df.empty:
        log.info("no fixtures in the feed for divisions %s", divisions)
        return {}

    # A fixture already in the past is stale feed data, not something to
    # predict. Drop it rather than insert a row that can never resolve.
    before = len(df)
    df = df[df["Date"].dt.date >= today]
    skipped_past = before - len(df)

    df["home_canon"] = df["HomeTeam"].map(canonical_team_name)
    df["away_canon"] = df["AwayTeam"].map(canonical_team_name)

    counts: dict[str, int] = {}
    with engine.begin() as conn:
        for division, group in df.groupby("Div"):
            names = set(group["home_canon"]) | set(group["away_canon"])
            ids = _team_ids(conn, names, division)
            inserted = 0
            for _, row in group.iterrows():
                match_date = row["Date"].date()
                params = {
                    "division": division,
                    "season": season_code(match_date),
                    "match_date": match_date,
                    "home_team_id": ids[row["home_canon"]],
                    "away_team_id": ids[row["away_canon"]],
                }
                # DO NOTHING, emphatically not DO UPDATE. The results ingester
                # upserts goals and result. If this used that pattern it would
                # write NULLs over a real result whenever a fixture row and a
                # played match collided on the key, destroying the data the
                # backtest depends on.
                res = conn.execute(
                    text(
                        """
                        INSERT INTO matches (division, season, match_date,
                                             home_team_id, away_team_id)
                        VALUES (:division, :season, :match_date,
                                :home_team_id, :away_team_id)
                        ON CONFLICT (division, season, match_date,
                                     home_team_id, away_team_id) DO NOTHING
                        """
                    ),
                    params,
                )
                inserted += res.rowcount or 0
            counts[division] = inserted
            log.info("%s: %s new fixtures", division, inserted)

    # No odds are written here. The feed carries opening prices only, and
    # storing those would put a weaker number where the benchmark expects a
    # closing line. Odds arrive later with the result, from the season file.
    if skipped_past:
        counts["skipped_past"] = skipped_past
    return counts


def stale_fixtures(engine: Engine | None = None, today: date | None = None) -> int:
    """Count unplayed matches whose date has already passed.

    These accumulate from postponements: the feed gives a date, the match moves,
    and the result lands under the new one. Harmless, but they must not be
    predicted -- predict_upcoming filters on date, and this reports the backlog.
    """
    engine = engine or get_engine()
    today = today or date.today()
    with engine.connect() as conn:
        return conn.execute(
            text(
                "SELECT COUNT(*) FROM matches "
                "WHERE result IS NULL AND match_date < :today"
            ),
            {"today": today},
        ).scalar_one()


def teams_without_history(
    engine: Engine | None = None, today: date | None = None
) -> list[str]:
    """Teams in scheduled fixtures that have never appeared in a played match.

    Two very different things land here and the code cannot tell them apart: a
    genuinely promoted club, which the models handle badly but honestly, and an
    unresolved name alias, which is a bug. "Atl. Madrid" in the fixtures feed
    against "Ath Madrid" in the results files split one club in two and yielded
    a 49.7% home probability for a side the market had near 77% -- confident,
    well-formed, and meaningless.

    Ingest deliberately does not raise on these, because a promoted club must
    not break the pipeline. So this surfaces them instead, and the caller
    reports them loudly enough that an alias gap does not go unnoticed.
    """
    engine = engine or get_engine()
    today = today or date.today()
    sql = text(
        """
        SELECT DISTINCT t.name
        FROM matches f
        JOIN teams t ON t.id IN (f.home_team_id, f.away_team_id)
        WHERE f.result IS NULL AND f.match_date >= :today
          AND NOT EXISTS (
              SELECT 1 FROM matches p
              WHERE p.result IS NOT NULL
                AND t.id IN (p.home_team_id, p.away_team_id)
          )
        ORDER BY t.name
        """
    )
    with engine.connect() as conn:
        return [r[0] for r in conn.execute(sql, {"today": today}).fetchall()]


def prune_stale_fixtures(engine: Engine | None = None, today: date | None = None) -> int:
    """Delete unplayed past matches and any predictions made for them.

    Not run automatically. Deleting a match discards what was predicted for it,
    and that record is the project's evidence of having forecast forward, so
    discarding it is an explicit choice rather than a side effect of ingesting.
    """
    engine = engine or get_engine()
    today = today or date.today()
    params = {"today": today}
    stale = "SELECT id FROM matches WHERE result IS NULL AND match_date < :today"
    with engine.begin() as conn:
        conn.execute(text(f"DELETE FROM predictions WHERE match_id IN ({stale})"), params)
        deleted = conn.execute(
            text("DELETE FROM matches WHERE result IS NULL AND match_date < :today"),
            params,
        )
    return deleted.rowcount or 0
