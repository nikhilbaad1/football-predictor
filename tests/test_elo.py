from __future__ import annotations

import pytest

from fpp.models.elo import DEFAULT_RATING, EloConfig, EloModel


class TestEloUpdate:
    def test_equal_ratings_expect_half_plus_home_edge(self):
        m = EloModel(config=EloConfig(home_advantage=0.0))
        assert m.expected_score("A", "B") == pytest.approx(0.5)

        m2 = EloModel(config=EloConfig(home_advantage=65.0))
        assert m2.expected_score("A", "B") > 0.5

    def test_expected_score_matches_closed_form(self):
        """400 points of advantage is by construction a 10:1 expectation."""
        m = EloModel(config=EloConfig(home_advantage=0.0))
        m.ratings = {"A": 1900.0, "B": 1500.0}
        assert m.expected_score("A", "B") == pytest.approx(10 / 11, abs=1e-6)

    def test_update_is_zero_sum(self):
        m = EloModel(config=EloConfig(home_advantage=0.0, mov_factor=False))
        m.ratings = {"A": 1500.0, "B": 1500.0}
        before = m.rating("A") + m.rating("B")
        m.update("A", "B", 3, 0)
        assert m.rating("A") + m.rating("B") == pytest.approx(before)

    def test_winner_gains_loser_loses(self):
        m = EloModel(config=EloConfig(mov_factor=False))
        m.ratings = {"A": 1500.0, "B": 1500.0}
        m.update("A", "B", 2, 0)
        assert m.rating("A") > 1500.0
        assert m.rating("B") < 1500.0

    def test_upset_moves_more_than_expected_result(self):
        strong_wins = EloModel(config=EloConfig(home_advantage=0.0, mov_factor=False))
        strong_wins.ratings = {"S": 1900.0, "W": 1500.0}
        strong_wins.update("S", "W", 1, 0)
        expected_gain = strong_wins.rating("S") - 1900.0

        upset = EloModel(config=EloConfig(home_advantage=0.0, mov_factor=False))
        upset.ratings = {"S": 1900.0, "W": 1500.0}
        upset.update("W", "S", 1, 0)
        upset_gain = upset.rating("W") - 1500.0

        assert upset_gain > expected_gain

    def test_margin_of_victory_scales_but_is_damped(self):
        def gain(hg, ag):
            m = EloModel(config=EloConfig(home_advantage=0.0, mov_factor=True))
            m.ratings = {"A": 1500.0, "B": 1500.0}
            m.update("A", "B", hg, ag)
            return m.rating("A") - 1500.0

        g1, g3, g5 = gain(1, 0), gain(3, 0), gain(5, 0)
        assert g3 > g1
        assert g5 > g3
        # Damped: a 5-0 must not be worth five times a 1-0.
        assert g5 < 5 * g1

    def test_draw_between_equals_changes_nothing(self):
        m = EloModel(config=EloConfig(home_advantage=0.0, mov_factor=False))
        m.ratings = {"A": 1500.0, "B": 1500.0}
        m.update("A", "B", 1, 1)
        assert m.rating("A") == pytest.approx(1500.0)


class TestEloFitting:
    def test_ranks_teams_correctly(self, synthetic_matches):
        m = EloModel().fit(synthetic_matches)
        assert m.rating("Strong FC") > m.rating("Good United")
        assert m.rating("Good United") > m.rating("Average City")
        assert m.rating("Average City") > m.rating("Poor Rovers")
        assert m.rating("Poor Rovers") > m.rating("Weak Town")

    def test_predict_proba_sums_to_one_and_is_hda(self, synthetic_matches):
        m = EloModel().fit(synthetic_matches)
        p = m.predict_proba("Strong FC", "Weak Town")
        assert p.sum() == pytest.approx(1.0)
        assert p[0] > p[2], "strong home side should be favoured"

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            EloModel().predict_proba("A", "B")

    def test_draw_probability_falls_as_gap_widens(self, synthetic_matches):
        """A fitted mapping should learn that mismatches draw less often."""
        m = EloModel().fit(synthetic_matches)
        close = m.predict_proba("Average City", "Good United")[1]
        lopsided = m.predict_proba("Strong FC", "Weak Town")[1]
        assert lopsided < close

    def test_season_regression_pulls_toward_mean(self):
        m = EloModel(config=EloConfig(regress_to_mean=0.5))
        m.ratings = {"A": 1700.0, "B": 1300.0}
        m.apply_season_regression()
        assert m.rating("A") == pytest.approx(1600.0)
        assert m.rating("B") == pytest.approx(1400.0)

    def test_unrated_team_gets_default(self):
        assert EloModel().rating("Nobody") == DEFAULT_RATING


class TestNoLeakage:
    def test_ratings_only_use_past_matches(self, synthetic_matches):
        """Fitting on a prefix must give the same ratings as fitting on the
        whole set would have given at that point — i.e. later matches cannot
        influence earlier ratings."""
        cut = 300
        prefix = EloModel().fit(synthetic_matches.iloc[:cut], fit_outcome_model=False)
        recorded = dict(prefix.ratings)

        full = EloModel()
        for row in synthetic_matches.iloc[:cut].itertuples(index=False):
            full.update(row.home_team, row.away_team, int(row.home_goals), int(row.away_goals))

        for team, rating in recorded.items():
            assert full.rating(team) == pytest.approx(rating)
