"""Mini-Project 1: UBOS District Population Forecaster.

Classes
-------
DistrictPopulation        validated container for one district's yearly series.
CAGRForecaster            exponential growth at the compound annual growth rate.
FibonacciRatioForecaster  previous cohort's method: scale by successive Fibonacci ratios.
ClassroomPlanner          turns a population change into additional classrooms.

The linear-trend model and the abstract ``Forecaster`` base class live in
``src.common.forecasting`` because Mini-Project 5 reuses them.

How the pieces fit together
---------------------------
1. ``DistrictPopulation`` holds and validates the data and answers descriptive
   questions (stats, YoY growth, CAGR) and can ``split`` itself into train/test.
2. Each forecaster (linear, CAGR, Fibonacci) is a ``Forecaster`` subclass, so
   ``compare_models`` can score them all the same way on the 2022-2024 holdout.
3. The winning model is passed to ``bootstrap_interval`` to get a point forecast
   for 2025-2029 plus a 95 % prediction interval.
4. ``ClassroomPlanner`` turns the 2024 -> 2029 population change into classrooms.

All populations are in thousands, as in the brief.
"""

from __future__ import annotations

import copy
import math
import statistics

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from src.common.forecasting import Forecaster, LinearTrendForecaster, holdout_evaluate

__all__ = [
    "DistrictPopulation", "CAGRForecaster", "FibonacciRatioForecaster",
    "LinearTrendForecaster", "ClassroomPlanner", "compare_models", "bootstrap_interval",
    "fibonacci_ratios",
]


class DistrictPopulation:
    """Population estimates (in thousands) for one district over consecutive years.

    Validation: years and populations must be equal-length, non-empty, numeric,
    populations must be non-negative and years strictly increasing by one (the
    growth-rate and forecasting maths assume an evenly spaced yearly series).
    """

    def __init__(self, name: str, years: ArrayLike, populations: ArrayLike) -> None:
        if not name or not str(name).strip():
            raise ValueError("district name must not be empty")
        yrs = np.asarray(years, dtype=int)
        pops = np.asarray(populations, dtype=float)
        if yrs.ndim != 1 or pops.ndim != 1:
            raise ValueError("years and populations must be one-dimensional")
        if yrs.size != pops.size:
            raise ValueError(f"years ({yrs.size}) and populations ({pops.size}) differ in length")
        if yrs.size == 0:
            raise ValueError("at least one observation is required")
        if not np.all(np.isfinite(pops)):
            raise ValueError("populations must be finite numbers")
        if np.any(pops < 0):
            raise ValueError("populations cannot be negative")
        if yrs.size > 1 and not np.all(np.diff(yrs) == 1):
            raise ValueError("years must be consecutive and increasing")
        self.name = str(name).strip()
        self.years = yrs
        self.populations = pops

    def __len__(self) -> int:
        return int(self.years.size)

    def __repr__(self) -> str:
        return (f"DistrictPopulation(name={self.name!r}, years={self.years[0]}-{self.years[-1]}, "
                f"n={len(self)}, latest={self.populations[-1]:,.0f}k)")

    # ----- descriptive statistics ------------------------------------------
    def stats_with_statistics(self) -> dict[str, float]:
        """Mean, median, *sample* variance and std via the ``statistics`` module."""
        data = self.populations.tolist()
        return {"mean": statistics.mean(data), "median": statistics.median(data),
                "variance": statistics.variance(data), "stdev": statistics.stdev(data)}

    def stats_with_numpy(self, ddof: int = 0) -> dict[str, float]:
        """Same statistics via NumPy. ``ddof=0`` (NumPy's default) is the population
        variance (divide by n); ``ddof=1`` is the sample variance (divide by n-1)."""
        p = self.populations
        return {"mean": float(np.mean(p)), "median": float(np.median(p)),
                "variance": float(np.var(p, ddof=ddof)), "stdev": float(np.std(p, ddof=ddof))}

    # ----- growth ------------------------------------------------------------
    def yoy_growth(self) -> np.ndarray:
        """Year-on-year growth rates (as fractions), length ``len(self) - 1``."""
        if len(self) < 2:
            raise ValueError("need at least two years to compute growth")
        if np.any(self.populations[:-1] == 0):
            raise ValueError("growth rate undefined when a population is zero")
        # np.diff gives P_t - P_{t-1}; dividing by the previous year's value
        # (populations[:-1] = every value except the last) gives the growth rate.
        return np.diff(self.populations) / self.populations[:-1]

    def cagr(self) -> float:
        """Compound annual growth rate: (last/first)^(1/(n-1)) - 1."""
        if len(self) < 2:
            raise ValueError("need at least two years to compute CAGR")
        first, last = self.populations[0], self.populations[-1]
        if first <= 0:
            raise ValueError("CAGR undefined when the first value is zero")
        # 10 yearly values contain 9 growth steps, hence the exponent 1/(n-1).
        # CAGR is the single constant rate g with first * (1+g)^(n-1) == last.
        return float((last / first) ** (1 / (len(self) - 1)) - 1)

    def split(self, last_train_year: int) -> tuple["DistrictPopulation", "DistrictPopulation"]:
        """Split into (train, test) with ``last_train_year`` as the final training year."""
        mask = self.years <= last_train_year
        if mask.all() or not mask.any():
            raise ValueError("split year must leave data on both sides")
        return (DistrictPopulation(self.name, self.years[mask], self.populations[mask]),
                DistrictPopulation(self.name, self.years[~mask], self.populations[~mask]))


class CAGRForecaster(Forecaster):
    """Exponential growth: ``y_t = y_0 * (1 + g)^t`` with g the training-period CAGR.

    Forecasts continue from the last observed value: ``y_{n-1} * (1+g)^k``.
    """

    name = "CAGR / exponential"
    min_obs = 2

    def __init__(self) -> None:
        super().__init__()
        self.growth_rate: float = np.nan

    def _fit(self, y: np.ndarray) -> None:
        if y[0] <= 0 or y[-1] < 0:
            raise ValueError("CAGR needs a positive starting value")
        self.growth_rate = float((y[-1] / y[0]) ** (1 / (y.size - 1)) - 1)

    def _predict(self, horizon: int) -> np.ndarray:
        k = np.arange(1, horizon + 1)          # 1, 2, ..., horizon years ahead
        # Start from the last *observed* value, not the fitted curve, so the
        # forecast joins the data without a jump.
        return self._y[-1] * (1 + self.growth_rate) ** k

    def fitted_values(self) -> np.ndarray:
        super().fitted_values()
        return self._y[0] * (1 + self.growth_rate) ** np.arange(self._y.size)


def fibonacci_ratios(count: int, start: int = 1) -> np.ndarray:
    """Return ``count`` successive ratios F(k+1)/F(k) beginning at k = ``start``.

    With F(1)=F(2)=1: start=1 -> 1, 2, 1.5, 1.667, 1.6, ... -> golden ratio 1.618.
    """
    if count < 1 or start < 1:
        raise ValueError("count and start must be positive")
    # Build just enough Fibonacci numbers: each new term is the sum of the two before it.
    fib = [1, 1]
    while len(fib) < start + count + 1:
        fib.append(fib[-1] + fib[-2])
    # Python lists start at 0, so fib[i] is F(i+1); ratio F(k+1)/F(k) = fib[k] / fib[k-1]
    return np.array([fib[k] / fib[k - 1] for k in range(start, start + count)])


class FibonacciRatioForecaster(Forecaster):
    """Previous cohort's model: multiply the last value by successive Fibonacci ratios.

    Forecast k is ``y_last * r_1 * r_2 * ... * r_k`` where r are Fibonacci ratios.
    Our interpretation: the ratio sequence starts at index ``start`` (default
    ``start=2``, so the ratios are 2, 1.5, 1.667, 1.6, ...). Whatever the start, the
    ratios converge to the golden ratio (~1.618), i.e. ~62% growth per year. The
    model has no parameters it learns from data, so it has no in-sample fit.
    """

    name = "Fibonacci ratio"
    min_obs = 1

    def __init__(self, start: int = 2) -> None:
        super().__init__()
        self.start = start

    def _fit(self, y: np.ndarray) -> None:
        pass  # nothing is estimated: the model only uses the last value

    def _predict(self, horizon: int) -> np.ndarray:
        # cumprod multiplies the ratios together step by step:
        # [r1, r1*r2, r1*r2*r3, ...], so forecast k = last value x product of k ratios.
        return self._y[-1] * np.cumprod(fibonacci_ratios(horizon, self.start))


class ClassroomPlanner:
    """Converts a population change into the number of extra classrooms needed.

    additional classrooms = ceil(max(0, growth in people) x school-age share / class size)
    We round *up* because a fraction of a classroom still has to be built.
    """

    def __init__(self, school_age_share: float = 0.18, pupils_per_classroom: int = 53) -> None:
        if not 0 < school_age_share <= 1:
            raise ValueError("school_age_share must be in (0, 1]")
        if pupils_per_classroom <= 0:
            raise ValueError("pupils_per_classroom must be positive")
        self.school_age_share = school_age_share
        self.pupils_per_classroom = pupils_per_classroom

    def extra_pupils(self, current_thousands: float, future_thousands: float) -> float:
        """Additional primary-school-age children between the two population levels."""
        if current_thousands < 0 or future_thousands < 0:
            raise ValueError("population cannot be negative")
        growth_people = max(0.0, future_thousands - current_thousands) * 1000
        return growth_people * self.school_age_share

    def additional_classrooms(self, current_thousands: float, future_thousands: float) -> int:
        """Classrooms to add; zero if the population does not grow."""
        return math.ceil(self.extra_pupils(current_thousands, future_thousands) / self.pupils_per_classroom)

    def __repr__(self) -> str:
        return f"ClassroomPlanner(share={self.school_age_share}, class_size={self.pupils_per_classroom})"


def compare_models(district: DistrictPopulation, models: list[Forecaster],
                   last_train_year: int) -> pd.DataFrame:
    """Holdout comparison table (MAE, RMSE, MAPE) for one district, best model first."""
    if not models:
        raise ValueError("at least one model is required")
    # Summing a boolean mask counts the True values, i.e. the number of training years.
    n_train = int(np.sum(district.years <= last_train_year))
    rows = []
    for m in models:
        res = holdout_evaluate(m, district.populations, n_train)
        rows.append({"model": m.name, "MAE": res["MAE"], "RMSE": res["RMSE"], "MAPE (%)": res["MAPE"]})
    return pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)


def bootstrap_interval(model: Forecaster, y: ArrayLike, horizon: int, n_boot: int = 1000,
                       level: float = 0.95, seed: int = 2026) -> dict[str, np.ndarray]:
    """Residual-bootstrap prediction interval.

    1. Fit the model and compute residuals e = y - fitted, centred to mean zero.
    2. Repeat ``n_boot`` times: build a synthetic history fitted + resampled e,
       refit the model on it, forecast, and add a resampled residual to every
       forecast step (so we capture both parameter and observation noise).
    3. Take the empirical (1-level)/2 and (1+level)/2 quantiles.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    y = np.asarray(y, dtype=float)
    base = copy.deepcopy(model).fit(y)
    fitted = base.fitted_values()
    if np.any(np.isnan(fitted)):
        raise ValueError(f"{model.name} has no in-sample fit, so residuals cannot be bootstrapped")
    # Centre the residuals: CAGR passes exactly through the first and last points, so its
    # raw residuals are one-sided and would shift every bootstrap path in one direction.
    resid = (y - fitted) - np.mean(y - fitted)
    rng = np.random.default_rng(seed)
    sims = np.empty((n_boot, horizon))
    for b in range(n_boot):
        # (a) a plausible alternative history: same trend, reshuffled noise
        y_star = fitted + rng.choice(resid, size=y.size, replace=True)
        # (b) refit on it, so the spread includes uncertainty in the model's parameters
        m = copy.deepcopy(model).fit(y_star)
        # (c) forecast and add fresh noise for each future year (observation uncertainty)
        sims[b] = m.predict(horizon) + rng.choice(resid, size=horizon, replace=True)
    # For a 95 % interval keep the middle 95 % of the simulated paths: 2.5th to 97.5th percentile.
    alpha = (1 - level) / 2
    return {"point": base.predict(horizon),
            "lower": np.quantile(sims, alpha, axis=0),
            "upper": np.quantile(sims, 1 - alpha, axis=0)}
