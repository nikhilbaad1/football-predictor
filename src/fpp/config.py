"""Settings. Everything configurable lives here, read once at import."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is convenience, not a requirement
    pass

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.getenv("DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR / 'football.db'}")

# football-data.co.uk division codes for the top-5 leagues.
# We train on all of these (ADR 0004) but display only DISPLAY_DIVISION.
DIVISIONS = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "I1": "Serie A",
    "D1": "Bundesliga",
    "F1": "Ligue 1",
}
DISPLAY_DIVISION = "E0"

FIRST_SEASON_START = 2016


def season_code(d: date) -> str:
    """The season a date falls in, named the way football-data.co.uk names it.

    Their season runs July to June, so 2026-08-20 and 2027-01-10 are both
    "2627". This matters more than it looks: the fixtures feed has no season
    column, while the results files are named by season and the matches table
    is keyed on it. Derive the code differently from the source and a fixture
    and its own result end up as two rows that never reconcile.
    """
    start = d.year if d.month >= 7 else d.year - 1
    return f"{start % 100:02d}{(start + 1) % 100:02d}"


CURRENT_SEASON = season_code(date.today())

# Generated rather than hand-listed. A hardcoded list silently stops including
# the current season once the calendar rolls over, and fixtures ingested for a
# season whose results file is never fetched would stay unplayed forever.
_CURRENT_START = date.today().year if date.today().month >= 7 else date.today().year - 1
DEFAULT_SEASONS = [
    season_code(date(y, 8, 1)) for y in range(FIRST_SEASON_START, _CURRENT_START + 1)
]

BASE_URL = "https://www.football-data.co.uk/mmz4281"

# Fixtures for roughly the next week, all divisions in one file. Separate from
# BASE_URL because it is a different shape: no season in the path, no results,
# and no closing odds (see ingest/fixtures.py).
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

# Outcome ordering. Used everywhere; never reorder without changing the DB too.
OUTCOMES = ("H", "D", "A")

for _d in (DATA_DIR, RAW_DIR):
    _d.mkdir(parents=True, exist_ok=True)
