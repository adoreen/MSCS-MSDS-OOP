"""Abstract forecasting interface shared by Mini-Projects 1 and 5.

What this module provides
-------------------------
* ``Forecaster``             abstract base class every forecasting model inherits from.
* ``LinearTrendForecaster``  straight-line trend, used by both P1 and P5.
* ``holdout_evaluate``       train on the first part of a series, test on the rest (P1).
* ``walk_forward``           rolling-origin one-step-ahead backtest (P5).

How the class design works
--------------------------
This is the *template method* pattern. The public methods ``fit`` and
``predict`` are defined once, here. They do the checks that every model needs
(1-D data, no NaNs, enough observations, a positive horizon) and then call
``_fit`` / ``_predict``, which each subclass implements with its own maths.

So a new model only has to answer two questions: "what do I learn from the
data?" (``_fit``) and "what do I predict next?" (``_predict``). It inherits
all the validation for free, and the backtesting functions below work with any
model without knowing which one it is (polymorphism).

Typical use::

    model = LinearTrendForecaster().fit([10, 12, 14, 16])
    model.predict(2)          # -> array([18., 20.])
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike

from src.common.metrics import mae, mape, rmse


class Forecaster(ABC):
    """Base class for univariate forecasters on an evenly spaced series.

    Subclasses must implement ``_fit`` and ``_predict`` and may override
    ``fitted_values`` and ``min_obs``.
    """

    #: Human-readable model name used in tables and plot legends.
    name: str = "Forecaster"
    #: Minimum number of observations needed to fit the model.
    min_obs: int = 1

    def __init__(self) -> None:
        self._y: np.ndarray | None = None

    # ----- public API -------------------------------------------------------
    def fit(self, y: ArrayLike) -> "Forecaster":
        """Fit the model to the observed series ``y`` and return ``self``."""
        arr = np.asarray(y, dtype=float)
        if arr.ndim != 1:
            raise ValueError("y must be one-dimensional")
        if arr.size < self.min_obs:
            raise ValueError(
                f"{self.name} needs at least {self.min_obs} observations, got {arr.size}"
            )
        if not np.all(np.isfinite(arr)):
            raise ValueError("y contains NaN or infinite values")
        self._y = arr
        self._fit(arr)
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """Forecast the next ``horizon`` values after the end of the training data."""
        if self._y is None:
            raise RuntimeError(f"{self.name} must be fitted before predict()")
        if not isinstance(horizon, (int, np.integer)) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        return np.asarray(self._predict(int(horizon)), dtype=float)

    def fitted_values(self) -> np.ndarray:
        """In-sample fitted values. Models without a natural fit return NaNs."""
        if self._y is None:
            raise RuntimeError(f"{self.name} must be fitted first")
        return np.full(self._y.size, np.nan)

    @property
    def is_fitted(self) -> bool:
        return self._y is not None

    def __repr__(self) -> str:
        state = "fitted" if self.is_fitted else "unfitted"
        return f"{type(self).__name__}({state})"

    # ----- hooks for subclasses --------------------------------------------
    @abstractmethod
    def _fit(self, y: np.ndarray) -> None:
        """Estimate model parameters from validated data."""

    @abstractmethod
    def _predict(self, horizon: int) -> np.ndarray:
        """Return ``horizon`` forecasts."""


class LinearTrendForecaster(Forecaster):
    """Straight-line trend ``y = a + b*t`` fitted by least squares (``np.polyfit``).

    ``t`` is the position in the series (0, 1, 2, ...). For yearly data this is
    equivalent to regressing on the year itself, just better conditioned.
    """

    name = "Linear trend"
    min_obs = 2

    def __init__(self) -> None:
        super().__init__()
        self.slope: float = np.nan
        self.intercept: float = np.nan

    def _fit(self, y: np.ndarray) -> None:
        t = np.arange(y.size)  # 0, 1, 2, ... one step per observation
        # polyfit with deg=1 solves the least-squares problem min sum (y - (a + b t))^2
        # and returns the coefficients highest power first: [b, a].
        self.slope, self.intercept = (float(c) for c in np.polyfit(t, y, deg=1))

    def _predict(self, horizon: int) -> np.ndarray:
        t_future = np.arange(self._y.size, self._y.size + horizon)
        return self.intercept + self.slope * t_future

    def fitted_values(self) -> np.ndarray:
        super().fitted_values()  # raises if unfitted
        return self.intercept + self.slope * np.arange(self._y.size)


def holdout_evaluate(model: Forecaster, y: ArrayLike, n_train: int) -> dict[str, object]:
    """Train on the first ``n_train`` points, forecast the rest, and score the forecast.

    A fresh copy of ``model`` is fitted so the caller's object is not changed.
    Returns a dict with the fitted model, predictions and MAE/RMSE/MAPE.
    """
    arr = np.asarray(y, dtype=float)
    if not 0 < n_train < arr.size:
        raise ValueError("n_train must leave at least one point for testing")
    # deepcopy gives an independent, unfitted copy, so evaluating a model never
    # changes the object the caller passed in (and one model object can be reused).
    fitted = copy.deepcopy(model).fit(arr[:n_train])
    test = arr[n_train:]
    pred = fitted.predict(test.size)
    return {
        "model": fitted,
        "predictions": pred,
        "MAE": mae(test, pred),
        "RMSE": rmse(test, pred),
        "MAPE": mape(test, pred),
    }


def walk_forward(model: Forecaster, y: ArrayLike, first_target: int) -> np.ndarray:
    """Rolling-origin (walk-forward) one-step-ahead forecasts.

    For every index ``t`` from ``first_target`` to the end, a fresh copy of the
    model is fitted on ``y[:t]`` and asked for one step ahead. Returns the
    predictions for ``y[first_target:]``. This mimics how the model would
    actually be used: only past data is ever seen.
    """
    arr = np.asarray(y, dtype=float)
    if not 0 < first_target < arr.size:
        raise ValueError("first_target must be inside the series and leave a training window")
    preds = []
    for t in range(first_target, arr.size):
        history = arr[:t]                       # everything before day t, nothing after
        step_model = copy.deepcopy(model).fit(history)
        preds.append(step_model.predict(1)[0])  # forecast for day t only
    return np.array(preds)
