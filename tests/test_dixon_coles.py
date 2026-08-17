"""Dixon-Coles correctness.

These check the maths against closed-form values and against known ground truth
in the synthetic data — not against stored outputs. A failure here should always
be explainable as "the model is now wrong about X", never "a number moved".
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import approx_fprime

from fpp.models.dixon_coles import (
    DixonColesConfig,
    DixonColesModel,
    _neg_log_likelihood,
    _neg_log_likelihood_grad,
    _tau,
)
from tests.conftest import TRUE_HOME_ADV, TRUE_STRENGTHS


class TestTauCorrection:
    """tau is the whole point of Dixon-Coles over plain Poisson."""

    def test_matches_paper_definition(self):
        lam = np.array([1.5, 1.5, 1.5, 1.5, 1.5])
        mu = np.array([1.2, 1.2, 1.2, 1.2, 1.2])
        x = np.array([0, 0, 1, 1, 3], dtype=float)
        y = np.array([0, 1, 0, 1, 2], dtype=float)
        rho = -0.1

        tau, *_ = _tau(lam, mu, x, y, rho)

        assert tau[0] == pytest.approx(1 - 1.5 * 1.2 * rho)  # (0,0)
        assert tau[1] == pytest.approx(1 + 1.5 * rho)        # (0,1)
        assert tau[2] == pytest.approx(1 + 1.2 * rho)        # (1,0)
        assert tau[3] == pytest.approx(1 - rho)              # (1,1)
        assert tau[4] == pytest.approx(1.0)                  # untouched

    def test_rho_zero_is_independent_poisson(self):
        lam = np.full(4, 1.4)
        mu = np.full(4, 1.1)
        x = np.array([0, 0, 1, 1], dtype=float)
        y = np.array([0, 1, 0, 1], dtype=float)
        tau, *_ = _tau(lam, mu, x, y, 0.0)
        assert np.allclose(tau, 1.0)

    def test_negative_rho_lifts_low_scores(self):
        """Negative rho is what makes 0-0 and 1-0 more likely, which is the
        empirically observed direction in football."""
        lam, mu, rho = np.array([1.4]), np.array([1.1]), -0.1
        t00, *_ = _tau(lam, mu, np.array([0.0]), np.array([0.0]), rho)
        t11, *_ = _tau(lam, mu, np.array([1.0]), np.array([1.0]), rho)
        assert t00[0] > 1.0   # 0-0 boosted
        assert t11[0] > 1.0   # 1-1 also boosted at this rho sign
        t01, *_ = _tau(lam, mu, np.array([0.0]), np.array([1.0]), rho)
        assert t01[0] < 1.0   # 0-1 suppressed


class TestGradient:
    def test_analytic_gradient_matches_numeric(self):
        """The analytic gradient exists for speed; if it is wrong the optimizer
        silently converges somewhere useless."""
        rng = np.random.default_rng(0)
        n = 4
        hi = np.array([0, 1, 2, 3, 0, 1])
        ai = np.array([1, 2, 3, 0, 2, 3])
        x = np.array([1, 0, 2, 1, 0, 3], dtype=float)
        y = np.array([0, 0, 1, 1, 1, 2], dtype=float)
        w = np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5])
        params = np.concatenate([rng.normal(0, 0.3, n), rng.normal(0, 0.3, n), [0.25], [-0.08]])

        analytic = _neg_log_likelihood_grad(params, hi, ai, x, y, w, n)
        numeric = approx_fprime(
            params, lambda p: _neg_log_likelihood(p, hi, ai, x, y, w, n), 1e-7
        )
        assert np.allclose(analytic, numeric, atol=1e-4), (
            f"max diff {np.abs(analytic - numeric).max():.2e}"
        )


class TestFitting:
    def test_recovers_known_strengths(self, synthetic_matches):
        """Data generated from known strengths should fit back to them."""
        model = DixonColesModel(
            config=DixonColesConfig(half_life_days=10_000)  # effectively unweighted
        ).fit(synthetic_matches)
        assert model.converged

        mean_strength = np.mean(list(TRUE_STRENGTHS.values()))
        true_centred = {t: s - mean_strength for t, s in TRUE_STRENGTHS.items()}
        fitted = np.array([model.attack[t] for t in TRUE_STRENGTHS])
        truth = np.array([true_centred[t] for t in TRUE_STRENGTHS])

        # Attack is only half the strength effect (defence carries the rest),
        # so check the ordering is right and the correlation is strong.
        assert np.corrcoef(fitted, truth)[0, 1] > 0.9
        assert model.attack["Strong FC"] > model.attack["Weak Town"]
        assert model.defence["Strong FC"] > model.defence["Weak Town"]

    def test_recovers_home_advantage(self, synthetic_matches):
        model = DixonColesModel(config=DixonColesConfig(half_life_days=10_000)).fit(
            synthetic_matches
        )
        assert model.home_advantage == pytest.approx(TRUE_HOME_ADV, abs=0.12)

    def test_empty_input_raises(self, synthetic_matches):
        with pytest.raises(ValueError):
            DixonColesModel().fit(synthetic_matches.iloc[:0])


@pytest.fixture(scope="module")
def fitted(synthetic_matches):
    return DixonColesModel().fit(synthetic_matches)


class TestPrediction:
    def test_probabilities_sum_to_one(self, fitted):
        p = fitted.predict_proba("Strong FC", "Weak Town")
        assert p.sum() == pytest.approx(1.0)
        assert (p >= 0).all()

    def test_ordering_is_home_draw_away(self, fitted):
        """Ordering is a convention the whole codebase depends on."""
        strong_home = fitted.predict_proba("Strong FC", "Weak Town")
        weak_home = fitted.predict_proba("Weak Town", "Strong FC")
        assert strong_home[0] > strong_home[2]
        assert weak_home[2] > weak_home[0]

    def test_home_advantage_is_real(self, fitted):
        """The same fixture reversed should favour whoever is at home."""
        a = fitted.predict_proba("Average City", "Good United")
        b = fitted.predict_proba("Good United", "Average City")
        assert a[0] > b[2]  # Average City at home beats Average City away

    def test_scoreline_matrix_normalized(self, fitted):
        m = fitted.scoreline_matrix("Strong FC", "Average City")
        assert m.sum() == pytest.approx(1.0)
        assert (m >= 0).all()

    def test_1x2_agrees_with_scoreline_matrix(self, fitted):
        """The three-way probabilities must be sums over the same matrix."""
        m = fitted.scoreline_matrix("Good United", "Poor Rovers")
        p = fitted.predict_proba("Good United", "Poor Rovers")
        assert p[0] == pytest.approx(np.tril(m, -1).sum())
        assert p[1] == pytest.approx(np.trace(m))
        assert p[2] == pytest.approx(np.triu(m, 1).sum())

    def test_unknown_team_falls_back_to_average(self, fitted):
        p = fitted.predict_proba("Newly Promoted", "Average City")
        assert p.sum() == pytest.approx(1.0)
        assert np.isfinite(p).all()
