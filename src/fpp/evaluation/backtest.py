"""Walk-forward backtest.

The one invariant that matters: a prediction for a match on date D is produced
by a model fitted only on matches strictly before D. Violating that inflates
every metric and invalidates the whole exercise, so it is enforced structurally
here rather than left to the caller to remember.

Refitting is periodic rather than per-match. Dixon-Coles takes a second or two
to fit on ~17k matches, so refitting for each of ~4,000 test matches would take
hours for no benefit — the parameters barely move over a handful of matches.
`refit_every` controls the trade-off; the model is still never fitted on data
from after the block it predicts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fpp.evaluation.metrics import base_rate_baseline, remove_vig, summarize
from fpp.models.blend import blend
from fpp.models.dixon_coles import DixonColesConfig, DixonColesModel
from fpp.models.elo import EloConfig, EloModel

log = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    predictions: pd.DataFrame       # per-match probabilities from every model
    metrics: pd.DataFrame           # one row per model
    n_train_initial: int
    n_test: int

    def __str__(self) -> str:
        cols = ["model", "n", "log_loss", "brier", "ece", "accuracy"]
        return self.metrics[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}")


def walk_forward(
    matches: pd.DataFrame,
    min_train: int = 2000,
    refit_every: int = 40,
    elo_config: EloConfig | None = None,
    dc_config: DixonColesConfig | None = None,
    blend_weights: tuple[float, float] = (0.35, 0.65),
    test_filter: pd.Series | None = None,
) -> BacktestResult:
    """Run the backtest.

    Args:
        matches: played matches, ascending by date, with the columns produced
            by `db.load_matches`.
        min_train: matches to accumulate before predicting anything.
        refit_every: refit cadence, in matches.
        blend_weights: (elo, dixon_coles). Dixon-Coles carries more weight
            because it models scorelines directly; the split is a prior, not a
            fitted value — see models/blend.py.
        test_filter: optional boolean Series aligned to `matches`, restricting
            which matches are *scored*. Training still uses everything, which is
            the point of pooled multi-league training (ADR 0004): train on five
            leagues, evaluate on one.
    """
    matches = matches.sort_values(["match_date", "match_id"], kind="stable").reset_index(drop=True)
    if len(matches) <= min_train:
        raise ValueError(f"need more than {min_train} matches, got {len(matches)}")

    rows: list[dict] = []
    elo_model: EloModel | None = None
    dc_model: DixonColesModel | None = None

    for start in range(min_train, len(matches), refit_every):
        train = matches.iloc[:start]
        block = matches.iloc[start : start + refit_every]
        if block.empty:
            break

        # Anchor time weighting at the first match of the block: on that date,
        # every training match is genuinely in the past.
        ref_date = block["match_date"].iloc[0]

        elo_model = EloModel(config=elo_config or EloConfig()).fit(train)
        try:
            dc_model = DixonColesModel(config=dc_config or DixonColesConfig()).fit(
                train, reference_date=ref_date
            )
        except Exception as exc:  # a failed fit should not abort the run
            log.warning("Dixon-Coles fit failed at %s: %s", ref_date, exc)
            dc_model = None

        p_elo = elo_model.predict_frame(block)
        p_dc = dc_model.predict_frame(block) if dc_model else p_elo
        p_blend = blend([p_elo, p_dc], list(blend_weights))

        for i, match in enumerate(block.itertuples(index=False)):
            rows.append(
                {
                    "match_id": match.match_id,
                    "match_date": match.match_date,
                    "division": match.division,
                    "home_team": match.home_team,
                    "away_team": match.away_team,
                    "result": match.result,
                    "elo_H": p_elo[i, 0], "elo_D": p_elo[i, 1], "elo_A": p_elo[i, 2],
                    "dc_H": p_dc[i, 0], "dc_D": p_dc[i, 1], "dc_A": p_dc[i, 2],
                    "blend_H": p_blend[i, 0], "blend_D": p_blend[i, 1], "blend_A": p_blend[i, 2],
                    "odds_home": match.odds_home,
                    "odds_draw": match.odds_draw,
                    "odds_away": match.odds_away,
                }
            )

        if start % (refit_every * 20) == 0:
            log.info("backtest %s/%s", start, len(matches))

    preds = pd.DataFrame(rows)

    if test_filter is not None:
        keep = set(matches.loc[test_filter.reindex(matches.index, fill_value=False), "match_id"])
        preds = preds[preds["match_id"].isin(keep)].reset_index(drop=True)

    outcomes = preds["result"].to_numpy()
    metrics = [
        summarize(preds[["elo_H", "elo_D", "elo_A"]].to_numpy(), outcomes, "elo"),
        summarize(preds[["dc_H", "dc_D", "dc_A"]].to_numpy(), outcomes, "dixon_coles"),
        summarize(preds[["blend_H", "blend_D", "blend_A"]].to_numpy(), outcomes, "blend"),
        summarize(base_rate_baseline(outcomes), outcomes, "baseline_base_rate"),
    ]

    book = remove_vig(preds[["odds_home", "odds_draw", "odds_away"]].to_numpy())
    valid = np.isfinite(book).all(axis=1)
    if valid.sum() > 50:
        metrics.append(summarize(book[valid], outcomes[valid], "baseline_bookmaker"))
        # Same subset, so the comparison against the bookmaker is like for like.
        metrics.append(
            summarize(
                preds.loc[valid, ["blend_H", "blend_D", "blend_A"]].to_numpy(),
                outcomes[valid],
                "blend (odds subset)",
            )
        )

    return BacktestResult(
        predictions=preds,
        metrics=pd.DataFrame(metrics),
        n_train_initial=min_train,
        n_test=len(preds),
    )
