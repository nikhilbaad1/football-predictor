"""Shared fixtures.

Synthetic data is generated from *known* team strengths so tests can assert
that the model recovers something close to the truth. Testing against a
snapshot of real results would only prove the code still does what it did
yesterday, not that it is correct.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

TRUE_STRENGTHS = {
    "Strong FC": 0.55,
    "Good United": 0.25,
    "Average City": 0.0,
    "Poor Rovers": -0.30,
    "Weak Town": -0.50,
}
TRUE_HOME_ADV = 0.28
BASE_RATE = 1.35


def _generate(n_rounds: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = list(TRUE_STRENGTHS)
    rows = []
    day = date(2018, 8, 1)
    match_id = 0

    for _ in range(n_rounds):
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                lam = BASE_RATE * np.exp(
                    TRUE_STRENGTHS[home] - TRUE_STRENGTHS[away] + TRUE_HOME_ADV
                )
                mu = BASE_RATE * np.exp(TRUE_STRENGTHS[away] - TRUE_STRENGTHS[home])
                hg, ag = int(rng.poisson(lam)), int(rng.poisson(mu))
                rows.append(
                    {
                        "match_id": match_id,
                        "division": "TEST",
                        "season": "9999",
                        "match_date": day,
                        "home_team": home,
                        "away_team": away,
                        "home_goals": hg,
                        "away_goals": ag,
                        "result": "H" if hg > ag else ("D" if hg == ag else "A"),
                        "odds_home": np.nan,
                        "odds_draw": np.nan,
                        "odds_away": np.nan,
                    }
                )
                match_id += 1
                day += timedelta(days=1)
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def synthetic_matches() -> pd.DataFrame:
    """~1,000 matches from known Poisson strengths."""
    return _generate(n_rounds=50)


@pytest.fixture(scope="session")
def small_matches() -> pd.DataFrame:
    return _generate(n_rounds=8, seed=7)
