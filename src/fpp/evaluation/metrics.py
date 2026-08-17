"""Scoring rules and the bookmaker baseline.

Accuracy is deliberately absent as a headline metric. A model that predicts the
favourite every time scores well on accuracy and is useless — what matters is
whether the probabilities are right, which is what proper scoring rules measure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

OUTCOMES = ("H", "D", "A")
_EPS = 1e-15


def _onehot(outcomes: np.ndarray) -> np.ndarray:
    return np.column_stack([(outcomes == o).astype(float) for o in OUTCOMES])


def log_loss(probas: np.ndarray, outcomes: np.ndarray) -> float:
    """Multiclass log loss. Lower is better. Baselines for football 1X2:
    ~1.099 for a uniform 1/3 guess, ~1.03 for always predicting base rates,
    ~0.96-0.98 for de-vigged closing odds in a top league."""
    p = np.clip(probas, _EPS, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    return float(-np.mean(np.log((p * _onehot(outcomes)).sum(axis=1))))


def brier_score(probas: np.ndarray, outcomes: np.ndarray) -> float:
    """Multiclass Brier: mean squared error over the three probabilities.
    Ranges 0 (perfect) to 2 (maximally wrong)."""
    return float(np.mean(((probas - _onehot(outcomes)) ** 2).sum(axis=1)))


def accuracy(probas: np.ndarray, outcomes: np.ndarray) -> float:
    """Reported for context only — see module docstring."""
    return float(np.mean(np.array(OUTCOMES)[probas.argmax(axis=1)] == outcomes))


def remove_vig(odds: np.ndarray) -> np.ndarray:
    """Convert decimal odds to probabilities, stripping the bookmaker margin.

    Uses the multiplicative method: invert each price, then divide by the
    overround. This is the standard simple approach and is what we benchmark
    against. It is known to be slightly biased — it removes margin evenly
    across outcomes, whereas real books load more onto longshots, so it mildly
    overstates the true probability of unlikely outcomes. Shin's method
    corrects for this and is worth revisiting if the baseline comparison ever
    becomes load-bearing.

    Rows with missing or invalid prices come back as NaN.
    """
    odds = np.asarray(odds, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        implied = 1.0 / odds
        total = implied.sum(axis=1, keepdims=True)
        probs = implied / total
    bad = ~np.isfinite(probs).all(axis=1) | (odds <= 1.0).any(axis=1)
    probs[bad] = np.nan
    return probs


def base_rate_baseline(outcomes: np.ndarray) -> np.ndarray:
    """Constant prediction at the observed outcome frequencies.

    Note this peeks at the full set of outcomes, so it is a slightly optimistic
    baseline — which is the point. If a model cannot beat a constant that was
    handed the answer key, it has learned nothing.
    """
    rates = np.array([(outcomes == o).mean() for o in OUTCOMES])
    return np.tile(rates, (len(outcomes), 1))


def calibration_table(probas: np.ndarray, outcomes: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Predicted vs observed frequency, pooled across all three outcomes.

    A well-calibrated model has predicted ~= observed in every bin. Systematic
    predicted > observed means overconfidence.
    """
    flat_p = probas.ravel()
    flat_y = _onehot(outcomes).ravel()
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(flat_p, edges) - 1, 0, bins - 1)

    rows = []
    for b in range(bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}",
                "n": int(mask.sum()),
                "predicted": float(flat_p[mask].mean()),
                "observed": float(flat_y[mask].mean()),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty:
        df["gap"] = df["predicted"] - df["observed"]
    return df


def expected_calibration_error(probas: np.ndarray, outcomes: np.ndarray, bins: int = 10) -> float:
    """Weighted mean absolute gap between predicted and observed. 0 is perfect."""
    table = calibration_table(probas, outcomes, bins)
    if table.empty:
        return float("nan")
    w = table["n"] / table["n"].sum()
    return float((w * table["gap"].abs()).sum())


def summarize(probas: np.ndarray, outcomes: np.ndarray, label: str = "model") -> dict:
    return {
        "model": label,
        "n": int(len(outcomes)),
        "log_loss": log_loss(probas, outcomes),
        "brier": brier_score(probas, outcomes),
        "ece": expected_calibration_error(probas, outcomes),
        "accuracy": accuracy(probas, outcomes),
    }
