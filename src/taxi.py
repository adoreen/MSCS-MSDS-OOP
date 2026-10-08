"""Mini-Project 5: Taxi (matatu) Route Revenue, Pricing & Fleet Planner.

Classes
-------
Route                       passenger counts + fare, revenue and descriptive statistics.
MarketEquilibrium           linear supply/demand solved as a 2x2 system.
MovingAverageForecaster     the original 3-day moving average.
SimpleExponentialSmoothing  SES with tunable alpha.
SeasonalNaiveForecaster     "same day last week" (extension).
FleetPlanner                vehicles needed for a forecast passenger count.

``LinearTrendForecaster`` and the ``Forecaster`` base class are shared with
Mini-Project 1 (``src.common.forecasting``).

Workflow in the notebook: ``Route`` describes each route, ``MarketEquilibrium``
answers the pricing question, every forecaster is scored with ``backtest_mae``
(walk-forward, days 4-10), ``tune_alpha`` picks alpha for SES, the best model
forecasts day 11, and ``FleetPlanner`` converts that forecast into vehicles.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike
from scipy import linalg

from src.common.forecasting import Forecaster, LinearTrendForecaster, walk_forward
from src.common.metrics import mae

__all__ = [
    "Route", "MarketEquilibrium", "MovingAverageForecaster", "SimpleExponentialSmoothing",
    "SeasonalNaiveForecaster", "LinearTrendForecaster", "FleetPlanner", "backtest_mae",
    "tune_alpha", "simulate_weekly_demand",
]


class Route:
    """One route's daily passenger counts and its (flat) fare in UGX."""

    def __init__(self, name: str, passengers: ArrayLike, fare: float) -> None:
        p = np.asarray(passengers, dtype=float)
        if not name:
            raise ValueError("route name required")
        if p.ndim != 1 or p.size == 0:
            raise ValueError("passengers must be a non-empty 1-D sequence")
        if np.any(p < 0) or not np.all(np.isfinite(p)):
            raise ValueError("passenger counts must be finite and non-negative")
        if fare <= 0:
            raise ValueError("fare must be positive")
        self.name = name
        self.passengers = p
        self.fare = float(fare)

    def __len__(self) -> int:
        return int(self.passengers.size)

    def __repr__(self) -> str:
        return f"Route({self.name!r}, days={len(self)}, fare=UGX {self.fare:,.0f})"

    def daily_revenue(self) -> np.ndarray:
        """UGX per day = passengers x fare."""
        return self.passengers * self.fare

    def total_revenue(self) -> float:
        return float(self.daily_revenue().sum())

    def describe(self) -> dict[str, float]:
        """Mean, sample variance and sample std of passengers using ``statistics``."""
        data = self.passengers.tolist()
        if len(data) < 2:
            raise ValueError("need at least two days for a sample variance")
        return {"mean": statistics.mean(data), "variance": statistics.variance(data),
                "stdev": statistics.stdev(data)}


class MarketEquilibrium:
    """Linear demand Qd = a - b P and supply Qs = c + d P.

    Rearranged as a linear system in the unknowns (Q, P)::

        Q + b P = a
        Q - d P = c
    """

    def __init__(self, demand_intercept: float, demand_slope: float,
                 supply_intercept: float, supply_slope: float) -> None:
        if demand_slope <= 0 or supply_slope <= 0:
            raise ValueError("slopes must be positive (demand falls and supply rises with price)")
        self.a, self.b = demand_intercept, demand_slope
        self.c, self.d = supply_intercept, supply_slope

    def system(self) -> tuple[np.ndarray, np.ndarray]:
        # Q = a - bP  ->  1*Q + b*P = a      (demand row)
        # Q = c + dP  ->  1*Q - d*P = c      (supply row)
        A = np.array([[1.0, self.b], [1.0, -self.d]])
        rhs = np.array([self.a, self.c])
        return A, rhs

    def solve(self) -> tuple[float, float]:
        """Return the equilibrium (price P*, quantity Q*)."""
        A, rhs = self.system()
        q, p = linalg.solve(A, rhs)
        return float(p), float(q)

    def quantity_demanded(self, price: float) -> float:
        return self.a - self.b * price

    def quantity_supplied(self, price: float) -> float:
        return self.c + self.d * price

    def position(self, price: float) -> str:
        """Whether ``price`` is 'below', 'above' or 'at' equilibrium."""
        p_star, _ = self.solve()
        if math.isclose(price, p_star):
            return "at"
        return "below" if price < p_star else "above"


class MovingAverageForecaster(Forecaster):
    """Forecast = mean of the last ``window`` observations (flat for all horizons)."""

    def __init__(self, window: int = 3) -> None:
        if window < 1:
            raise ValueError("window must be at least 1")
        super().__init__()
        self.window = window
        self.min_obs = window
        self.name = f"{window}-day moving average"

    def _fit(self, y: np.ndarray) -> None:
        self.level = float(y[-self.window:].mean())

    def _predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self.level)


class SimpleExponentialSmoothing(Forecaster):
    """SES: level_t = alpha * y_t + (1 - alpha) * level_{t-1}, initial level = y_0.

    Forecasts are flat at the final level. alpha near 1 reacts quickly to new
    data; alpha near 0 averages over a long history.
    """

    def __init__(self, alpha: float = 0.5) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        super().__init__()
        self.alpha = alpha
        self.name = f"SES (alpha={alpha:.2f})"

    def _fit(self, y: np.ndarray) -> None:
        level = y[0]          # start the level at the first observation
        levels = [level]
        for value in y[1:]:
            # new level = alpha x today's value + (1 - alpha) x previous level.
            # Unrolled, this is a weighted average with weights alpha, alpha(1-alpha),
            # alpha(1-alpha)^2, ..., so older days count exponentially less.
            level = self.alpha * value + (1 - self.alpha) * level
            levels.append(level)
        self.levels = np.array(levels)
        self.level = float(level)

    def _predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self.level)


class SeasonalNaiveForecaster(Forecaster):
    """Forecast for day t+k = value one season (``period`` days) earlier."""

    def __init__(self, period: int = 7) -> None:
        if period < 1:
            raise ValueError("period must be at least 1")
        super().__init__()
        self.period = period
        self.min_obs = period
        self.name = f"Seasonal naive (period={period})"

    def _fit(self, y: np.ndarray) -> None:
        self.last_season = y[-self.period:].copy()

    def _predict(self, horizon: int) -> np.ndarray:
        # k % period cycles through the last week again and again: Mon, Tue, ..., Sun, Mon, ...
        return np.array([self.last_season[k % self.period] for k in range(horizon)])


class FleetPlanner:
    """Vehicles = ceil(passengers x (1 + buffer) / (trips per vehicle x seats))."""

    def __init__(self, trips_per_vehicle: int = 8, seats: int = 14, buffer: float = 0.15) -> None:
        if trips_per_vehicle <= 0 or seats <= 0:
            raise ValueError("trips and seats must be positive")
        if buffer < 0:
            raise ValueError("buffer cannot be negative")
        self.trips_per_vehicle = trips_per_vehicle
        self.seats = seats
        self.buffer = buffer

    @property
    def capacity_per_vehicle(self) -> int:
        """Passengers one vehicle can carry per day."""
        return self.trips_per_vehicle * self.seats

    def vehicles_needed(self, passengers: float) -> int:
        """Round *up*: a part-vehicle means some passengers would be left behind.
        At least one vehicle is deployed whenever any demand is expected."""
        if passengers < 0:
            raise ValueError("passengers cannot be negative")
        return math.ceil(passengers * (1 + self.buffer) / self.capacity_per_vehicle)

    def __repr__(self) -> str:
        return f"FleetPlanner(trips={self.trips_per_vehicle}, seats={self.seats}, buffer={self.buffer:.0%})"


def backtest_mae(model: Forecaster, y: ArrayLike, first_target: int = 3) -> float:
    """Walk-forward MAE. ``first_target=3`` (0-based) means forecasting days 4..end."""
    y = np.asarray(y, dtype=float)
    preds = walk_forward(model, y, first_target)
    return mae(y[first_target:], preds)


def tune_alpha(y: ArrayLike, grid: Sequence[float] | None = None,
               first_target: int = 3) -> tuple[float, float]:
    """Grid-search alpha for SES by walk-forward MAE. Returns (best alpha, its MAE)."""
    grid = np.round(np.arange(0.05, 1.0001, 0.05), 2) if grid is None else grid
    if len(grid) == 0:
        raise ValueError("alpha grid is empty")
    scores = {float(a): backtest_mae(SimpleExponentialSmoothing(float(a)), y, first_target) for a in grid}
    best = min(scores, key=scores.get)
    return best, scores[best]


def simulate_weekly_demand(days: int = 60, base: float = 50.0, seed: int = 11,
                           noise_sd: float = 3.0) -> np.ndarray:
    """Synthetic passenger counts with a weekly cycle (day 0 = Monday).

    Multipliers: Mon-Thu around 1.0, Friday 1.35 (people travel home / out),
    Saturday 1.1, Sunday 0.6 (quiet). Counts are rounded to whole passengers.
    """
    if days < 1:
        raise ValueError("days must be positive")
    pattern = np.array([1.0, 0.95, 1.0, 1.05, 1.35, 1.1, 0.6])   # Mon ... Sun
    rng = np.random.default_rng(seed)
    # np.arange(days) % 7 gives the weekday of each day, which picks its multiplier
    y = base * pattern[np.arange(days) % 7] + rng.normal(0, noise_sd, days)
    return np.round(np.clip(y, 0, None))
