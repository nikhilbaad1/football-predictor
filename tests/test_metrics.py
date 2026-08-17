from __future__ import annotations

import numpy as np
import pytest

from fpp.evaluation.metrics import (
    base_rate_baseline,
    brier_score,
    calibration_table,
    expected_calibration_error,
    log_loss,
    remove_vig,
)


class TestLogLoss:
    def test_uniform_guess_is_log_three(self):
        p = np.full((100, 3), 1 / 3)
        y = np.array(["H", "D", "A"] * 33 + ["H"])
        assert log_loss(p, y) == pytest.approx(np.log(3), abs=1e-9)

    def test_perfect_prediction_is_near_zero(self):
        y = np.array(["H", "D", "A"])
        p = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        assert log_loss(p, y) < 1e-10

    def test_confident_and_wrong_is_heavily_penalised(self):
        y = np.array(["A"])
        confident = log_loss(np.array([[0.98, 0.01, 0.01]]), y)
        hedged = log_loss(np.array([[0.4, 0.3, 0.3]]), y)
        assert confident > hedged

    def test_handles_zero_probability_without_infinity(self):
        y = np.array(["A"])
        assert np.isfinite(log_loss(np.array([[1.0, 0.0, 0.0]]), y))


class TestBrier:
    def test_perfect_is_zero(self):
        y = np.array(["H", "A"])
        p = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        assert brier_score(p, y) == pytest.approx(0.0)

    def test_maximally_wrong_is_two(self):
        y = np.array(["H"])
        p = np.array([[0.0, 0.0, 1.0]])
        assert brier_score(p, y) == pytest.approx(2.0)

    def test_uniform_guess_closed_form(self):
        """Uniform 1/3: (1-1/3)^2 + 2*(1/3)^2 = 4/9 + 2/9 = 2/3."""
        y = np.array(["H", "D", "A"])
        p = np.full((3, 3), 1 / 3)
        assert brier_score(p, y) == pytest.approx(2 / 3)


class TestRemoveVig:
    def test_strips_overround_and_normalizes(self):
        odds = np.array([[2.0, 4.0, 4.0]])  # implied 0.5 + 0.25 + 0.25 = 1.0, no margin
        p = remove_vig(odds)
        assert p.sum() == pytest.approx(1.0)
        assert p[0, 0] == pytest.approx(0.5)

    def test_removes_a_real_margin(self):
        odds = np.array([[1.9, 3.6, 3.8]])
        implied = (1 / odds).sum()
        assert implied > 1.0, "test fixture should have a margin"
        p = remove_vig(odds)
        assert p.sum() == pytest.approx(1.0)

    def test_preserves_ordering(self):
        odds = np.array([[1.5, 4.5, 7.0]])
        p = remove_vig(odds)[0]
        assert p[0] > p[1] > p[2]

    def test_invalid_odds_become_nan(self):
        bad = np.array([[np.nan, 3.0, 4.0], [1.0, 3.0, 4.0], [2.0, 4.0, 4.0]])
        p = remove_vig(bad)
        assert np.isnan(p[0]).all()
        assert np.isnan(p[1]).all(), "odds of 1.0 imply certainty and are invalid"
        assert np.isfinite(p[2]).all()


class TestBaselines:
    def test_base_rate_matches_observed_frequencies(self):
        y = np.array(["H"] * 45 + ["D"] * 25 + ["A"] * 30)
        p = base_rate_baseline(y)
        assert p[0, 0] == pytest.approx(0.45)
        assert p[0, 1] == pytest.approx(0.25)
        assert p[0, 2] == pytest.approx(0.30)

    def test_base_rate_beats_uniform_on_real_distribution(self):
        y = np.array(["H"] * 45 + ["D"] * 25 + ["A"] * 30)
        assert log_loss(base_rate_baseline(y), y) < log_loss(np.full((100, 3), 1 / 3), y)


class TestCalibration:
    def test_perfectly_calibrated_has_near_zero_error(self):
        rng = np.random.default_rng(0)
        n = 20_000
        p_home = rng.uniform(0.1, 0.8, n)
        remainder = 1 - p_home
        p = np.column_stack([p_home, remainder * 0.4, remainder * 0.6])

        draws = rng.random(n)
        y = np.where(
            draws < p[:, 0], "H", np.where(draws < p[:, 0] + p[:, 1], "D", "A")
        )
        assert expected_calibration_error(p, y) < 0.02

    def test_overconfidence_is_detected(self):
        """Claim 90% home wins, deliver 60%. The 0.9-1.0 bin should show it.

        Note the table pools all three outcome columns, so the *largest* bin
        here is the low-probability one holding the two 0.05 columns — the
        overconfidence lives specifically in the high bin, so assert on that.
        """
        rng = np.random.default_rng(1)
        n = 5_000
        p = np.tile([0.9, 0.05, 0.05], (n, 1))
        y = np.where(rng.random(n) < 0.6, "H", "A")

        table = calibration_table(p, y).set_index("bin")
        high = table.loc["0.9-1.0"]
        assert high["predicted"] == pytest.approx(0.9)
        assert high["observed"] == pytest.approx(0.6, abs=0.03)
        assert high["gap"] > 0.2, "predicted should exceed observed"

        assert expected_calibration_error(p, y) > 0.1
