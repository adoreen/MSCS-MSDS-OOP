"""Tests for Mini-Project 1 (population forecaster)."""

import statistics

import numpy as np
import pytest

from src.population import (CAGRForecaster, ClassroomPlanner, DistrictPopulation,
                            FibonacciRatioForecaster, bootstrap_interval, compare_models,
                            fibonacci_ratios)
from src.common.forecasting import LinearTrendForecaster

YEARS = np.arange(2015, 2025)
GULU = [320, 330, 345, 360, 375, 390, 410, 430, 455, 480]


@pytest.fixture
def gulu() -> DistrictPopulation:
    return DistrictPopulation("Gulu", YEARS, GULU)


def test_dunder_methods(gulu):
    assert len(gulu) == 10
    assert "Gulu" in repr(gulu) and "2015-2024" in repr(gulu)


@pytest.mark.parametrize("years, pops", [
    (YEARS, GULU[:-1]),                      # unequal lengths
    (YEARS, [-1] + GULU[1:]),                # negative value
    ([], []),                                # empty
    ([2015, 2017, 2018], [1, 2, 3]),         # gap in years
])
def test_invalid_input_rejected(years, pops):
    with pytest.raises(ValueError):
        DistrictPopulation("Bad", years, pops)


def test_statistics_vs_numpy_ddof(gulu):
    """statistics.variance (n-1) equals np.var only when ddof=1."""
    s = gulu.stats_with_statistics()
    assert s["variance"] == pytest.approx(gulu.stats_with_numpy(ddof=1)["variance"])
    pop = gulu.stats_with_numpy(ddof=0)["variance"]
    assert pop == pytest.approx(s["variance"] * 9 / 10)
    assert pop == pytest.approx(statistics.pvariance(GULU))


def test_growth_and_cagr(gulu):
    assert gulu.yoy_growth()[0] == pytest.approx(10 / 320)
    assert gulu.cagr() == pytest.approx(1.5 ** (1 / 9) - 1)  # 480/320 = 1.5 over 9 steps


def test_cagr_forecaster_reproduces_constant_growth():
    y = 100 * 1.05 ** np.arange(8)
    model = CAGRForecaster().fit(y)
    assert model.growth_rate == pytest.approx(0.05)
    np.testing.assert_allclose(model.predict(2), [y[-1] * 1.05, y[-1] * 1.05 ** 2])


def test_fibonacci_ratios_converge_to_golden_ratio():
    np.testing.assert_allclose(fibonacci_ratios(4, start=1), [1, 2, 1.5, 5 / 3])
    assert fibonacci_ratios(1, start=30)[0] == pytest.approx((1 + 5 ** 0.5) / 2)
    pred = FibonacciRatioForecaster(start=2).fit([100]).predict(2)
    np.testing.assert_allclose(pred, [200, 300])


def test_classroom_planner_rounds_up_and_handles_decline():
    planner = ClassroomPlanner(0.18, 53)
    # 100k extra people -> 18,000 pupils -> 339.6 classrooms -> 340
    assert planner.additional_classrooms(1000, 1100) == 340
    assert planner.additional_classrooms(500, 450) == 0       # shrinking district
    assert planner.additional_classrooms(500, 500) == 0       # zero growth
    with pytest.raises(ValueError):
        ClassroomPlanner(school_age_share=0)


def test_compare_models_and_bootstrap(gulu):
    table = compare_models(gulu, [LinearTrendForecaster(), CAGRForecaster(),
                                  FibonacciRatioForecaster()], last_train_year=2021)
    assert list(table.columns) == ["model", "MAE", "RMSE", "MAPE (%)"]
    assert table.iloc[-1]["model"] == "Fibonacci ratio"   # the explosive model is worst
    pi = bootstrap_interval(LinearTrendForecaster(), GULU, horizon=5, n_boot=200, seed=1)
    assert np.all(pi["lower"] <= pi["point"]) and np.all(pi["point"] <= pi["upper"])
    with pytest.raises(ValueError):
        bootstrap_interval(FibonacciRatioForecaster(), GULU, horizon=5)
