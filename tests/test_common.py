"""Tests for the shared metrics and forecasting base class."""

import numpy as np
import pytest

from src.common.forecasting import LinearTrendForecaster, holdout_evaluate, walk_forward
from src.common.metrics import mae, mape, rmse


def test_metrics_match_hand_calculation():
    actual, pred = [100, 200, 300], [110, 190, 330]
    # errors: -10, 10, -30 -> MAE = 50/3, RMSE = sqrt(1100/3), MAPE = (10% + 5% + 10%)/3
    assert mae(actual, pred) == pytest.approx(50 / 3)
    assert rmse(actual, pred) == pytest.approx(np.sqrt(1100 / 3))
    assert mape(actual, pred) == pytest.approx(25 / 3)


def test_metrics_reject_empty_mismatched_and_zero():
    with pytest.raises(ValueError):
        mae([], [])
    with pytest.raises(ValueError):
        rmse([1, 2], [1])
    with pytest.raises(ValueError):
        mape([0, 1], [1, 1])


def test_linear_trend_recovers_exact_line():
    y = 5 + 2 * np.arange(6)
    model = LinearTrendForecaster().fit(y)
    assert model.slope == pytest.approx(2)
    np.testing.assert_allclose(model.predict(3), [17, 19, 21])
    np.testing.assert_allclose(model.fitted_values(), y)


def test_forecaster_validation():
    with pytest.raises(RuntimeError):
        LinearTrendForecaster().predict(1)
    with pytest.raises(ValueError):
        LinearTrendForecaster().fit([1.0])           # needs 2 points
    with pytest.raises(ValueError):
        LinearTrendForecaster().fit([1, np.nan, 3])
    with pytest.raises(ValueError):
        LinearTrendForecaster().fit([1, 2, 3]).predict(0)


def test_holdout_and_walk_forward_do_not_modify_model():
    y = np.arange(10, dtype=float)
    model = LinearTrendForecaster()
    res = holdout_evaluate(model, y, 7)
    assert res["MAE"] == pytest.approx(0)
    np.testing.assert_allclose(walk_forward(model, y, 3), y[3:])
    assert not model.is_fitted
