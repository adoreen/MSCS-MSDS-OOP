"""Tests for Mini-Project 3 (fish stock & export risk)."""

import numpy as np
import pytest

from src.fisheries import FishStock, PriceModel, RiskAssessor, fibonacci, weekly_revenue


def test_fibonacci_baseline():
    assert fibonacci(7) == [1, 1, 2, 3, 5, 8, 13]
    assert fibonacci(0) == []
    with pytest.raises(ValueError):
        fibonacci(-1)


def test_logistic_step_matches_hand_calculation():
    stock = FishStock(r=0.4, K=10_000, N0=4_000, h=0.1)
    nxt, harvest = stock.step(4_000)
    # 4000 + 0.4*4000*(1 - 0.4) - 0.1*4000 = 4000 + 960 - 400
    assert nxt == pytest.approx(4_560)
    assert harvest == pytest.approx(400)


def test_simulation_converges_to_equilibrium_and_msy():
    stock = FishStock(h=0.2)
    assert stock.msy == pytest.approx(1_000)
    assert stock.equilibrium_stock() == pytest.approx(5_000)
    res = stock.simulate(500)
    assert res.final_stock == pytest.approx(5_000, rel=1e-6)
    assert len(res.stock) == 501 and len(res.harvest) == 500


def test_edge_cases_zero_stock_and_invalid_parameters():
    res = FishStock(N0=0).simulate(10)
    assert np.all(res.stock == 0)              # no fish -> no growth, no harvest
    assert FishStock(h=0.5).equilibrium_stock() == 0   # h >= r collapses the stock
    with pytest.raises(ValueError):
        FishStock(h=1.5)
    with pytest.raises(ValueError):
        FishStock(K=0)


def test_closed_season_stops_harvest():
    res = FishStock(closed_weeks=range(0, 8)).simulate(52)
    assert np.all(res.harvest[:8] == 0) and np.all(res.harvest[8:] > 0)


def test_price_model_is_bounded_and_seeded():
    pm = PriceModel(step_sd=2_000)
    paths = pm.simulate_paths(52, 200)
    assert paths.min() >= 9_000 and paths.max() <= 16_000
    np.testing.assert_array_equal(paths, pm.simulate_paths(52, 200))
    with pytest.raises(ValueError):
        PriceModel(start=20_000)


def test_revenue_units():
    # 2 tonnes at 10,000 UGX/kg = 20 million UGX
    assert weekly_revenue([2.0], [10_000])[0] == pytest.approx(20_000_000)


def test_risk_classification_and_var():
    ra = RiskAssessor(0.10, 0.25)
    assert ra.classify([100, 100, 101, 99]) == "Low"
    assert ra.classify([50, 150, 60, 140]) == "High"
    var = ra.value_at_risk(np.arange(1, 101))
    assert var["quantile"] == pytest.approx(5.95)
    with pytest.raises(ValueError):
        ra.value_at_risk([])
    with pytest.raises(ValueError):
        ra.summary([1.0])


def test_cv_is_scale_free_but_variance_is_not():
    """Why a fixed variance threshold is meaningless: rescaling the units changes it."""
    ugx = [1e6, 1.2e6, 0.9e6]
    thousands = [v / 1000 for v in ugx]
    s1, s2 = RiskAssessor.summary(ugx), RiskAssessor.summary(thousands)
    assert s1["cv"] == pytest.approx(s2["cv"])
    assert s1["variance"] == pytest.approx(s2["variance"] * 1e6)


def test_mean_path_cv_averages_over_paths():
    paths = np.array([[100.0, 100.0, 100.0], [50.0, 100.0, 150.0]])
    # path 1: CV 0; path 2: sd 50 / mean 100 = 0.5 -> average 0.25
    assert RiskAssessor.mean_path_cv(paths) == pytest.approx(0.25)
    assert RiskAssessor().classify_paths(paths) == "High"
    with pytest.raises(ValueError):
        RiskAssessor.mean_path_cv([[0.0, 0.0]])
