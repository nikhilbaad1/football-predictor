"""Dixon-Coles: time-weighted bivariate Poisson with low-score correction.

Dixon & Coles (1997), "Modelling Association Football Scores and Inefficiencies
in the Football Betting Market", JRSS-C 46(2).

Goals are modelled as Poisson with team-specific attack and defence strengths:

    lambda = exp(attack_home - defence_away + home_advantage)     home goals
    mu     = exp(attack_away - defence_home)                      away goals

Independent Poisson gets low scores wrong — it under-predicts 0-0 and 1-0 and
over-predicts 1-1. Dixon-Coles adds a correction tau applied to the four
scorelines where both teams score at most one:

    tau(0,0) = 1 - lambda*mu*rho
    tau(0,1) = 1 + lambda*rho
    tau(1,0) = 1 + mu*rho
    tau(1,1) = 1 - rho
    tau(x,y) = 1                    otherwise

with rho typically fitting around -0.1 for football.

Matches are exponentially down-weighted by age so recent form dominates.

Implementation notes:
  * The likelihood has one redundant degree of freedom: adding a constant to
    every attack and every defence leaves lambda and mu unchanged. We fit
    freely and re-centre attack to mean zero afterwards, which is exact.
  * An analytic gradient is supplied. With ~150 teams the parameter vector is
    ~300 long, and a numerical gradient would need ~300 likelihood evaluations
    per step — too slow to refit repeatedly inside a walk-forward backtest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

MAX_GOALS = 10  # scoreline matrix is (MAX_GOALS+1) square; P(>10 goals) is negligible


@dataclass
class DixonColesConfig:
    half_life_days: float = 180.0  # weight halves every ~6 months
    max_goals: int = MAX_GOALS
    rho_bounds: tuple[float, float] = (-0.25, 0.25)


@dataclass
class DixonColesModel:
    config: DixonColesConfig = field(default_factory=DixonColesConfig)
    teams: list[str] = field(default_factory=list)
    attack: dict[str, float] = field(default_factory=dict)
    defence: dict[str, float] = field(default_factory=dict)
    home_advantage: float = 0.0
    rho: float = 0.0
    converged: bool = False

    # ---------------------------------------------------------------- fitting

    def fit(self, matches: pd.DataFrame, reference_date: date | None = None) -> DixonColesModel:
        """Fit on played matches.

        `reference_date` anchors the time weighting and must be on or after the
        last training match — it is normally the date being predicted. Passing a
        date inside the training window would weight future matches most
        heavily, which is leakage.
        """
        if matches.empty:
            raise ValueError("no matches to fit")

        ref = reference_date or matches["match_date"].max()
        self.teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
        index = {t: i for i, t in enumerate(self.teams)}
        n = len(self.teams)

        hi = matches["home_team"].map(index).to_numpy()
        ai = matches["away_team"].map(index).to_numpy()
        x = matches["home_goals"].to_numpy(dtype=float)
        y = matches["away_goals"].to_numpy(dtype=float)

        age_days = np.array([(ref - d).days for d in matches["match_date"]], dtype=float)
        xi = np.log(2.0) / self.config.half_life_days
        w = np.exp(-xi * np.maximum(age_days, 0.0))

        # [attack (n) | defence (n) | home_advantage | rho]
        p0 = np.concatenate([np.zeros(n), np.zeros(n), [0.25], [-0.05]])
        bounds = [(-3, 3)] * n + [(-3, 3)] * n + [(-1, 2), self.config.rho_bounds]

        result = minimize(
            _neg_log_likelihood,
            p0,
            args=(hi, ai, x, y, w, n),
            jac=_neg_log_likelihood_grad,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-9},
        )

        att, dfn = result.x[:n], result.x[n : 2 * n]
        shift = att.mean()  # exact re-centring; lambda and mu are unchanged
        att, dfn = att - shift, dfn - shift

        self.attack = dict(zip(self.teams, att, strict=True))
        self.defence = dict(zip(self.teams, dfn, strict=True))
        self.home_advantage = float(result.x[2 * n])
        self.rho = float(result.x[2 * n + 1])
        self.converged = bool(result.success)
        return self

    # ------------------------------------------------------------- prediction

    def rates(self, home: str, away: str) -> tuple[float, float]:
        """Expected goals for (home, away). Unknown teams fall back to average."""
        ah, dh = self.attack.get(home, 0.0), self.defence.get(home, 0.0)
        aa, da = self.attack.get(away, 0.0), self.defence.get(away, 0.0)
        lam = float(np.exp(ah - da + self.home_advantage))
        mu = float(np.exp(aa - dh))
        return lam, mu

    def scoreline_matrix(self, home: str, away: str) -> np.ndarray:
        """P(home scores i, away scores j) as a square matrix, normalized."""
        lam, mu = self.rates(home, away)
        k = np.arange(self.config.max_goals + 1)
        matrix = np.outer(poisson.pmf(k, lam), poisson.pmf(k, mu))

        matrix[0, 0] *= 1.0 - lam * mu * self.rho
        matrix[0, 1] *= 1.0 + lam * self.rho
        matrix[1, 0] *= 1.0 + mu * self.rho
        matrix[1, 1] *= 1.0 - self.rho

        matrix = np.clip(matrix, 0.0, None)
        return matrix / matrix.sum()

    def predict_proba(self, home: str, away: str) -> np.ndarray:
        """P(home), P(draw), P(away) — always in that order."""
        m = self.scoreline_matrix(home, away)
        draw = float(np.trace(m))
        home_win = float(np.tril(m, -1).sum())  # rows = home goals, so below diagonal
        away_win = float(np.triu(m, 1).sum())
        return np.array([home_win, draw, away_win])

    def predict_frame(self, matches: pd.DataFrame) -> np.ndarray:
        return np.vstack([
            self.predict_proba(r.home_team, r.away_team)
            for r in matches.itertuples(index=False)
        ])

    def most_likely_score(self, home: str, away: str) -> tuple[int, int, float]:
        m = self.scoreline_matrix(home, away)
        i, j = np.unravel_index(int(np.argmax(m)), m.shape)
        return int(i), int(j), float(m[i, j])


# ------------------------------------------------------------- likelihood


def _tau(lam: np.ndarray, mu: np.ndarray, x: np.ndarray, y: np.ndarray, rho: float):
    """Dixon-Coles correction, and its partials wrt lambda, mu, rho."""
    tau = np.ones_like(lam)
    d_lam = np.zeros_like(lam)
    d_mu = np.zeros_like(lam)
    d_rho = np.zeros_like(lam)

    m00 = (x == 0) & (y == 0)
    m01 = (x == 0) & (y == 1)
    m10 = (x == 1) & (y == 0)
    m11 = (x == 1) & (y == 1)

    tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
    d_lam[m00] = -mu[m00] * rho
    d_mu[m00] = -lam[m00] * rho
    d_rho[m00] = -lam[m00] * mu[m00]

    tau[m01] = 1.0 + lam[m01] * rho
    d_lam[m01] = rho
    d_rho[m01] = lam[m01]

    tau[m10] = 1.0 + mu[m10] * rho
    d_mu[m10] = rho
    d_rho[m10] = mu[m10]

    tau[m11] = 1.0 - rho
    d_rho[m11] = -1.0

    return np.maximum(tau, 1e-10), d_lam, d_mu, d_rho


def _unpack(params: np.ndarray, hi, ai, n: int):
    att, dfn = params[:n], params[n : 2 * n]
    gamma, rho = params[2 * n], params[2 * n + 1]
    lam = np.exp(att[hi] - dfn[ai] + gamma)
    mu = np.exp(att[ai] - dfn[hi])
    return lam, mu, rho


def _neg_log_likelihood(params, hi, ai, x, y, w, n) -> float:
    lam, mu, rho = _unpack(params, hi, ai, n)
    tau, *_ = _tau(lam, mu, x, y, rho)
    ll = w * (np.log(tau) + x * np.log(lam) - lam + y * np.log(mu) - mu)
    return -float(ll.sum())


def _neg_log_likelihood_grad(params, hi, ai, x, y, w, n) -> np.ndarray:
    lam, mu, rho = _unpack(params, hi, ai, n)
    tau, dt_dlam, dt_dmu, dt_drho = _tau(lam, mu, x, y, rho)

    # d(loglik)/d(lambda) * d(lambda)/d(param); lambda = exp(...) so the chain
    # rule contributes a factor of lambda for attack/home_adv and -lambda for
    # the opposing defence.
    c_lam = w * (dt_dlam / tau + x / lam - 1.0) * lam
    c_mu = w * (dt_dmu / tau + y / mu - 1.0) * mu

    grad = np.zeros_like(params)
    np.add.at(grad, hi, c_lam)              # attack_home
    np.add.at(grad, n + ai, -c_lam)         # defence_away
    np.add.at(grad, ai, c_mu)               # attack_away
    np.add.at(grad, n + hi, -c_mu)          # defence_home
    grad[2 * n] = c_lam.sum()               # home advantage
    grad[2 * n + 1] = (w * dt_drho / tau).sum()
    return -grad
