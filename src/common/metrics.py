"""Forecast error metrics used by Mini-Projects 1 and 5.

All functions take ``actual`` and ``predicted`` sequences of the same length and
return a single float. They are written out by hand (rather than imported from
scikit-learn) because the assignment asks us to implement the methods ourselves.

With errors e_i = actual_i - predicted_i over n points:

* MAE  = (1/n) * sum |e_i|                  same units as the data, easy to explain
* RMSE = sqrt((1/n) * sum e_i^2)            squaring punishes big misses more
* MAPE = (100/n) * sum |e_i / actual_i|     unit-free %, but breaks if actual = 0

RMSE >= MAE always; a large gap between them means a few big errors dominate.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _as_pair(actual: ArrayLike, predicted: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Convert inputs to float arrays and check they are non-empty and aligned."""
    a = np.asarray(actual, dtype=float).ravel()
    p = np.asarray(predicted, dtype=float).ravel()
    if a.size == 0:
        raise ValueError("actual and predicted must not be empty")
    if a.shape != p.shape:
        raise ValueError(f"length mismatch: {a.size} actual vs {p.size} predicted")
    return a, p


def mae(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Mean Absolute Error: average size of the error, in the data's own units."""
    a, p = _as_pair(actual, predicted)
    return float(np.mean(np.abs(a - p)))


def rmse(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Root Mean Squared Error: like MAE but penalises large errors more heavily."""
    a, p = _as_pair(actual, predicted)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mape(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Mean Absolute Percentage Error, returned as a percentage (e.g. 3.2 means 3.2%).

    MAPE is undefined when any actual value is zero, so we raise instead of
    silently returning ``inf``.
    """
    a, p = _as_pair(actual, predicted)
    if np.any(a == 0):
        raise ValueError("MAPE is undefined when an actual value is zero")
    return float(np.mean(np.abs((a - p) / a)) * 100)
