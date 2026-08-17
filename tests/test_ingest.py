from __future__ import annotations

import pandas as pd
import pytest

from fpp.ingest.football_data_uk import _pick_odds
from fpp.ingest.teams import canonical_team_name


class TestTeamNormalization:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Man United", "Manchester United"),
            ("Man Utd", "Manchester United"),
            ("MAN UNITED", "Manchester United"),
            ("  Man United  ", "Manchester United"),
            ("Nott'm Forest", "Nottingham Forest"),
            ("Wolves", "Wolverhampton Wanderers"),
            ("Ath Madrid", "Atletico Madrid"),
            ("M'gladbach", "Borussia Monchengladbach"),
            ("Paris SG", "Paris Saint-Germain"),
            ("Inter", "Inter Milan"),
        ],
    )
    def test_known_aliases(self, raw, expected):
        assert canonical_team_name(raw) == expected

    def test_aliases_converge(self):
        """Different spellings of one club must land on one canonical name —
        this is what keeps a team's history from being split in two."""
        variants = ["Man United", "Man Utd", "Manchester Utd"]
        assert len({canonical_team_name(v) for v in variants}) == 1

    def test_unknown_passes_through_cleaned(self):
        """A newly promoted club must not break ingestion."""
        assert canonical_team_name("Newly  Promoted   FC") == "Newly Promoted FC"

    def test_canonical_name_is_idempotent(self):
        once = canonical_team_name("Man United")
        assert canonical_team_name(once) == once

    def test_none_raises(self):
        with pytest.raises(ValueError):
            canonical_team_name(None)


class TestOddsSelection:
    def test_prefers_market_average_over_single_book(self):
        row = pd.Series({
            "AvgCH": 2.0, "AvgCD": 3.5, "AvgCA": 4.0,
            "B365CH": 1.9, "B365CD": 3.4, "B365CA": 3.9,
        })
        source, h, d, a = _pick_odds(row)
        assert source == "avg_closing"
        assert (h, d, a) == (2.0, 3.5, 4.0)

    def test_falls_back_to_bet365(self):
        row = pd.Series({"B365CH": 1.9, "B365CD": 3.4, "B365CA": 3.9})
        source, h, _, _ = _pick_odds(row)
        assert source == "b365_closing"
        assert h == 1.9

    def test_ignores_opening_odds(self):
        """B365H is the OPENING price. Only the closing line (B365CH) counts —
        silently using opening odds would weaken the benchmark without failing."""
        row = pd.Series({"B365H": 1.9, "B365D": 3.4, "B365A": 3.9})
        assert _pick_odds(row) is None

    def test_missing_odds_returns_none(self):
        assert _pick_odds(pd.Series({"AvgCH": 2.0, "AvgCD": None, "AvgCA": 4.0})) is None

    def test_rejects_impossible_prices(self):
        assert _pick_odds(pd.Series({"AvgCH": 1.0, "AvgCD": 3.5, "AvgCA": 4.0})) is None


class TestDateParsing:
    def test_uk_format_is_parsed_day_first(self):
        """05/08/2024 is 5 August, not 8 May. Getting this wrong reorders the
        whole dataset and silently corrupts every walk-forward split."""
        parsed = pd.to_datetime(pd.Series(["05/08/2024"]), dayfirst=True).iloc[0]
        assert (parsed.day, parsed.month) == (5, 8)
