"""Tests for Mini-Project 5 (taxi routes)."""

import numpy as np
import pytest

from src.taxi import (FleetPlanner, MarketEquilibrium, MovingAverageForecaster, Route,
                      SeasonalNaiveForecaster, SimpleExponentialSmoothing, backtest_mae,
                      simulate_weekly_demand, tune_alpha)

NTINDA = [35, 40, 42, 50, 55, 60, 48, 52, 47, 45]


def test_route_revenue_and_stats():
    r = Route("Kampala-Ntinda", NTINDA, 2_000)
    assert r.daily_revenue()[0] == 70_000
    assert r.total_revenue() == sum(NTINDA) * 2_000
    assert r.describe()["mean"] == pytest.approx(47.4)
    assert len(r) == 10


@pytest.mark.parametrize("pax, fare", [([], 2000), ([10, -1], 2000), ([10, 20], 0)])
def test_route_rejects_invalid_input(pax, fare):
    with pytest.raises(ValueError):
        Route("Bad", pax, fare)


def test_equilibrium_matches_algebra():
    # 120 - 0.02P = 10 + 0.03P  ->  P* = 2200, Q* = 76
    m = MarketEquilibrium(120, 0.02, 10, 0.03)
    p, q = m.solve()
    assert p == pytest.approx(2_200) and q == pytest.approx(76)
    assert m.position(2_000) == "below"
    assert m.quantity_demanded(2_000) > m.quantity_supplied(2_000)   # excess demand


def test_forecasters_hand_values():
    assert MovingAverageForecaster(3).fit([1, 2, 3, 6]).predict(1)[0] == pytest.approx(11 / 3)
    # SES alpha=0.5 on [10, 20]: level = 0.5*20 + 0.5*10 = 15
    assert SimpleExponentialSmoothing(0.5).fit([10, 20]).predict(2).tolist() == [15, 15]
    assert SeasonalNaiveForecaster(3).fit([1, 2, 3, 4, 5, 6]).predict(4).tolist() == [4, 5, 6, 4]
    with pytest.raises(ValueError):
        MovingAverageForecaster(3).fit([1, 2])     # not enough history
    with pytest.raises(ValueError):
        SimpleExponentialSmoothing(0)


def test_backtest_and_alpha_tuning():
    flat = [50.0] * 10
    assert backtest_mae(MovingAverageForecaster(3), flat) == 0
    alpha, score = tune_alpha(NTINDA)
    assert 0 < alpha <= 1
    assert score <= backtest_mae(SimpleExponentialSmoothing(0.5), NTINDA) + 1e-12


def test_fleet_planner_rounds_up():
    fp = FleetPlanner(8, 14, 0.15)
    assert fp.capacity_per_vehicle == 112
    assert fp.vehicles_needed(100) == 2       # 115 / 112 -> 1.03 -> 2
    assert fp.vehicles_needed(0) == 0
    with pytest.raises(ValueError):
        fp.vehicles_needed(-1)


def test_simulated_demand_has_weekly_pattern():
    y = simulate_weekly_demand(60, seed=1)
    assert len(y) == 60
    fridays, sundays = y[4::7], y[6::7]
    assert fridays.mean() > sundays.mean()
    np.testing.assert_array_equal(y, simulate_weekly_demand(60, seed=1))
