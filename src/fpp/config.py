"""Settings. Everything configurable lives here, read once at import."""

from __future__ import annotations

import os
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

# Season codes as football-data.co.uk names them: 2024/25 -> "2425".
DEFAULT_SEASONS = [
    "1617", "1718", "1819", "1920", "2021",
    "2122", "2223", "2324", "2425", "2526",
]

BASE_URL = "https://www.football-data.co.uk/mmz4281"

# Outcome ordering. Used everywhere; never reorder without changing the DB too.
OUTCOMES = ("H", "D", "A")

for _d in (DATA_DIR, RAW_DIR):
    _d.mkdir(parents=True, exist_ok=True)
