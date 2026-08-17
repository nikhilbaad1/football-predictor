"""Produce predictions for upcoming fixtures.

Kept separate from the API so it can run as a scheduled job — the page should
read precomputed rows, never fit a model on request.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import pandas as pd

from fpp.config import DISPLAY_DIVISION, DIVISIONS
from fpp.db import load_matches
from fpp.models.blend import blend
from fpp.models.dixon_coles import DixonColesModel
from fpp.models.elo import EloModel

log = logging.getLogger(__name__)

MODEL_VERSION = "elo+dc-v0.1"
BLEND_WEIGHTS = (0.35, 0.65)


@dataclass
class Fixture:
    home_team: str
    away_team: str
    match_date: date | None = None


def fit_models(
    training: pd.DataFrame, reference_date: date | None = None
) -> tuple[EloModel, DixonColesModel]:
    """Fit both models on everything available up to `reference_date`."""
    ref = reference_date or date.today()
    training = training[training["match_date"] < ref]
    if training.empty:
        raise ValueError(f"no training matches before {ref}")

    elo = EloModel().fit(training)
    dc = DixonColesModel().fit(training, reference_date=ref)
    if not dc.converged:
        log.warning("Dixon-Coles did not converge — predictions may be unreliable")
    return elo, dc


def predict_fixtures(
    fixtures: list[Fixture],
    elo: EloModel,
    dc: DixonColesModel,
) -> pd.DataFrame:
    """Blended three-way probabilities plus the most likely scoreline."""
    frame = pd.DataFrame(
        [{"home_team": f.home_team, "away_team": f.away_team} for f in fixtures]
    )
    p = blend([elo.predict_frame(frame), dc.predict_frame(frame)], list(BLEND_WEIGHTS))

    rows = []
    for i, f in enumerate(fixtures):
        hg, ag, score_p = dc.most_likely_score(f.home_team, f.away_team)
        lam, mu = dc.rates(f.home_team, f.away_team)
        rows.append(
            {
                "match_date": f.match_date,
                "home_team": f.home_team,
                "away_team": f.away_team,
                "prob_home": round(float(p[i, 0]), 4),
                "prob_draw": round(float(p[i, 1]), 4),
                "prob_away": round(float(p[i, 2]), 4),
                "expected_goals_home": round(lam, 2),
                "expected_goals_away": round(mu, 2),
                "likeliest_score": f"{hg}-{ag}",
                "likeliest_score_prob": round(score_p, 4),
                "model_version": MODEL_VERSION,
            }
        )
    return pd.DataFrame(rows)


def predict_upcoming(division: str = DISPLAY_DIVISION) -> pd.DataFrame:
    """Predict every unplayed fixture in the DB for one division.

    Trains on the pooled top-5 leagues (ADR 0004) but returns only `division`.
    """
    played = load_matches(divisions=list(DIVISIONS), played_only=True)
    if played.empty:
        raise RuntimeError("no matches in the database — run scripts/ingest.py first")

    upcoming = load_matches(divisions=[division], played_only=False)
    upcoming = upcoming[upcoming["result"].isna()]

    if upcoming.empty:
        log.warning(
            "no unplayed fixtures for %s. football-data.co.uk publishes results, "
            "not forward fixtures — use scripts/predict.py with explicit team names, "
            "or add a fixtures source.",
            division,
        )
        return pd.DataFrame()

    elo, dc = fit_models(played)
    fixtures = [
        Fixture(r.home_team, r.away_team, r.match_date)
        for r in upcoming.itertuples(index=False)
    ]
    return predict_fixtures(fixtures, elo, dc)
