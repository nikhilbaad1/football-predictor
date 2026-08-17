"""Blend model outputs and calibrate.

Week 1-3 uses a fixed weighted average. That is deliberately the simplest thing
that works: a learned stack needs its own held-out data, and with ~400
validation matches, splitting further to fit blend weights costs more than the
weights are worth. Revisit when the pooled multi-league dataset is in place.

Calibration is separate from blending and matters more. A model can rank
matches well and still be systematically overconfident; isotonic regression
fixes the probabilities without touching the ranking.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def blend(probas: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """Weighted average of several (n, 3) probability arrays."""
    if not probas:
        raise ValueError("nothing to blend")
    shapes = {p.shape for p in probas}
    if len(shapes) != 1:
        raise ValueError(f"shape mismatch: {shapes}")

    w = np.ones(len(probas)) if weights is None else np.asarray(weights, dtype=float)
    if len(w) != len(probas):
        raise ValueError("weights and probas differ in length")
    w = w / w.sum()

    out = sum(wi * p for wi, p in zip(w, probas))
    return out / out.sum(axis=1, keepdims=True)


class ProbabilityCalibrator:
    """Per-outcome isotonic calibration, renormalized to sum to 1.

    Fit on out-of-sample predictions only. Fitting on the same predictions the
    model was trained on will report near-perfect calibration and mean nothing.
    """

    def __init__(self) -> None:
        self.models: list[IsotonicRegression] = []

    def fit(self, probas: np.ndarray, outcomes: np.ndarray) -> ProbabilityCalibrator:
        """`outcomes` is an (n,) array of 'H'/'D'/'A'."""
        self.models = []
        for i, label in enumerate(("H", "D", "A")):
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso.fit(probas[:, i], (outcomes == label).astype(float))
            self.models.append(iso)
        return self

    def transform(self, probas: np.ndarray) -> np.ndarray:
        if not self.models:
            raise RuntimeError("call fit() before transform()")
        out = np.column_stack([m.predict(probas[:, i]) for i, m in enumerate(self.models)])
        out = np.clip(out, 1e-6, None)
        return out / out.sum(axis=1, keepdims=True)
