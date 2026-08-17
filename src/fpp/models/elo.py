"""Elo ratings, plus a fitted mapping from rating difference to 1X2.

Elo on its own produces an *expected score* (a win counts 1, a draw 0.5), not
three probabilities. Splitting that expected score into home/draw/away needs a
separate assumption, and hand-waving it is a common way these implementations
go quietly wrong.

We fit it instead: a multinomial logistic regression on the rating difference,
trained only on matches before the prediction date. That keeps the draw
probability empirical rather than assumed, and lets the model learn that draws
get less likely as the rating gap widens.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from fpp.config import OUTCOMES

DEFAULT_RATING = 1500.0


@dataclass
class EloConfig:
    k: float = 20.0                # update weight
    home_advantage: float = 65.0   # rating points added to the home side
    mov_factor: bool = True        # scale updates by margin of victory
    regress_to_mean: float = 0.0   # 0 = off; e.g. 0.15 pulls 15% toward 1500 each season


@dataclass
class EloModel:
    """Sequential Elo. Ratings only ever advance forward through time.

    The no-leakage guarantee is structural: `fit` walks matches in date order
    and each rating reflects only earlier matches.
    """

    config: EloConfig = field(default_factory=EloConfig)
    ratings: dict[str, float] = field(default_factory=dict)
    _outcome_model: LogisticRegression | None = None
    _history: list[dict] = field(default_factory=list)

    def rating(self, team: str) -> float:
        return self.ratings.get(team, DEFAULT_RATING)

    def expected_score(self, home: str, away: str) -> float:
        """Expected points share for the home side, in [0, 1]."""
        diff = self.rating(home) + self.config.home_advantage - self.rating(away)
        return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

    def rating_diff(self, home: str, away: str) -> float:
        return self.rating(home) + self.config.home_advantage - self.rating(away)

    def update(self, home: str, away: str, home_goals: int, away_goals: int) -> None:
        expected = self.expected_score(home, away)
        actual = 1.0 if home_goals > away_goals else (0.5 if home_goals == away_goals else 0.0)

        k = self.config.k
        if self.config.mov_factor:
            # Margin-of-victory multiplier, damped so a 5-0 doesn't move the
            # rating five times as far as a 1-0. Log scaling is the common
            # choice; the +1 keeps a one-goal win at multiplier 1.
            k *= np.log1p(abs(home_goals - away_goals))  / np.log(2)

        delta = k * (actual - expected)
        self.ratings[home] = self.rating(home) + delta
        self.ratings[away] = self.rating(away) - delta

    def fit(self, matches: pd.DataFrame, fit_outcome_model: bool = True) -> EloModel:
        """Walk matches in date order, updating ratings and recording the
        pre-match rating difference for each (used to fit the 1X2 mapping)."""
        matches = matches.sort_values(["match_date", "match_id"], kind="stable")
        self._history.clear()

        for row in matches.itertuples(index=False):
            self._history.append(
                {"diff": self.rating_diff(row.home_team, row.away_team), "result": row.result}
            )
            self.update(row.home_team, row.away_team, int(row.home_goals), int(row.away_goals))

        if fit_outcome_model and self._history:
            hist = pd.DataFrame(self._history)
            X = hist[["diff"]].to_numpy()
            y = hist["result"].to_numpy()
            self._outcome_model = LogisticRegression(max_iter=1000, C=1.0)
            self._outcome_model.fit(X, y)
        return self

    def predict_proba(self, home: str, away: str) -> np.ndarray:
        """Return P(home), P(draw), P(away) — always in that order."""
        if self._outcome_model is None:
            raise RuntimeError("call fit() before predict_proba()")
        diff = np.array([[self.rating_diff(home, away)]])
        raw = self._outcome_model.predict_proba(diff)[0]
        # sklearn orders columns by sorted class label ('A','D','H'); we need HDA.
        order = [list(self._outcome_model.classes_).index(o) for o in OUTCOMES]
        return raw[order]

    def predict_frame(self, matches: pd.DataFrame) -> np.ndarray:
        return np.vstack([
            self.predict_proba(r.home_team, r.away_team)
            for r in matches.itertuples(index=False)
        ])

    def apply_season_regression(self) -> None:
        """Pull ratings toward the mean between seasons (promotion/relegation
        churn means last season's rating overstates what we know)."""
        alpha = self.config.regress_to_mean
        if alpha <= 0:
            return
        for team, rating in self.ratings.items():
            self.ratings[team] = rating + alpha * (DEFAULT_RATING - rating)
