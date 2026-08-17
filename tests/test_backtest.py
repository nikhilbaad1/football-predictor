"""Backtest tests.

The leakage test is the most important one in the suite. If it ever fails,
every metric the project reports is wrong, so it is written to fail loudly
rather than approximately.
"""

from __future__ import annotations

import numpy as np
import pytest

from fpp.evaluation.backtest import walk_forward
from fpp.models.blend import ProbabilityCalibrator, blend


@pytest.fixture(scope="module")
def result(synthetic_matches):
    return walk_forward(synthetic_matches, min_train=400, refit_every=100)


class TestWalkForward:
    def test_produces_predictions_for_every_test_match(self, result, synthetic_matches):
        assert result.n_test == len(synthetic_matches) - 400
        assert len(result.predictions) == result.n_test

    def test_all_probabilities_valid(self, result):
        for prefix in ("elo", "dc", "blend"):
            p = result.predictions[[f"{prefix}_H", f"{prefix}_D", f"{prefix}_A"]].to_numpy()
            assert np.allclose(p.sum(axis=1), 1.0), f"{prefix} does not sum to 1"
            assert (p >= 0).all()
            assert np.isfinite(p).all()

    def test_beats_uniform_guessing(self, result):
        """Data has real signal in it, so a working model must beat log(3)."""
        blend_ll = result.metrics.set_index("model").loc["blend", "log_loss"]
        assert blend_ll < np.log(3)

    def test_reports_a_base_rate_baseline(self, result):
        assert "baseline_base_rate" in set(result.metrics["model"])

    def test_too_few_matches_raises(self, small_matches):
        with pytest.raises(ValueError, match="need more than"):
            walk_forward(small_matches, min_train=10_000)


class TestNoLookahead:
    def test_shuffling_future_results_cannot_change_past_predictions(self, synthetic_matches):
        """The definitive leakage check.

        Predictions for the first test block depend only on matches before it.
        Corrupting every result *after* that block must therefore leave those
        predictions bit-for-bit identical. If they move, something is reading
        the future.
        """
        min_train, refit_every = 400, 100

        clean = walk_forward(synthetic_matches, min_train=min_train, refit_every=refit_every)
        first_block = clean.predictions.head(refit_every)

        corrupted = synthetic_matches.copy().reset_index(drop=True)
        tail = corrupted.index >= (min_train + refit_every)
        rng = np.random.default_rng(99)
        corrupted.loc[tail, "home_goals"] = rng.integers(0, 8, tail.sum())
        corrupted.loc[tail, "away_goals"] = rng.integers(0, 8, tail.sum())
        corrupted.loc[tail, "result"] = np.where(
            corrupted.loc[tail, "home_goals"] > corrupted.loc[tail, "away_goals"], "H",
            np.where(
                corrupted.loc[tail, "home_goals"] == corrupted.loc[tail, "away_goals"], "D", "A"
            ),
        )

        dirty = walk_forward(corrupted, min_train=min_train, refit_every=refit_every)
        dirty_first = dirty.predictions.head(refit_every)

        for col in ("blend_H", "blend_D", "blend_A"):
            np.testing.assert_allclose(
                first_block[col].to_numpy(),
                dirty_first[col].to_numpy(),
                rtol=1e-12,
                err_msg=f"{col} changed when only FUTURE results were altered — leakage",
            )


class TestBlend:
    def test_output_is_normalized(self):
        a = np.array([[0.5, 0.3, 0.2]])
        b = np.array([[0.3, 0.4, 0.3]])
        out = blend([a, b], [0.5, 0.5])
        assert out.sum() == pytest.approx(1.0)
        assert out[0, 0] == pytest.approx(0.4)

    def test_weights_are_respected(self):
        a = np.array([[1.0, 0.0, 0.0]])
        b = np.array([[0.0, 0.0, 1.0]])
        assert blend([a, b], [1.0, 0.0])[0, 0] == pytest.approx(1.0)
        assert blend([a, b], [0.0, 1.0])[0, 2] == pytest.approx(1.0)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            blend([np.zeros((2, 3)), np.zeros((3, 3))])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            blend([])


class TestCalibrator:
    def test_fixes_systematic_overconfidence(self):
        rng = np.random.default_rng(3)
        n = 4_000
        true_p = rng.uniform(0.2, 0.7, n)
        y = np.where(rng.random(n) < true_p, "H", "A")

        # Overconfident: push predictions away from the middle.
        skewed = np.clip(true_p * 1.4, 0.01, 0.98)
        raw = np.column_stack([skewed, (1 - skewed) * 0.3, (1 - skewed) * 0.7])
        raw = raw / raw.sum(axis=1, keepdims=True)

        from fpp.evaluation.metrics import expected_calibration_error

        before = expected_calibration_error(raw, y)
        calibrated = ProbabilityCalibrator().fit(raw, y).transform(raw)
        after = expected_calibration_error(calibrated, y)

        assert after < before
        assert np.allclose(calibrated.sum(axis=1), 1.0)

    def test_transform_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            ProbabilityCalibrator().transform(np.full((2, 3), 1 / 3))
