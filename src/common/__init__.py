"""Code shared by more than one mini-project (forecasting base class, metrics)."""

from src.common.metrics import mae, mape, rmse
from src.common.forecasting import Forecaster, LinearTrendForecaster

__all__ = ["mae", "rmse", "mape", "Forecaster", "LinearTrendForecaster"]
