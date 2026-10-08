"""Mini-Project 3: Lake Victoria Fish Stock & Export Risk Model (Jinja cooperative).

Classes
-------
FishStock     discrete logistic growth with proportional harvesting (+ optional closed season).
PriceModel    bounded, seeded random walk for the weekly fish price (UGX/kg).
RiskAssessor  descriptive statistics, CV-based risk class and Monte Carlo Value-at-Risk.

Units: stock and harvest in tonnes, price in UGX/kg, revenue in UGX.

Where the theory comes from
---------------------------
At a steady state N(t+1) = N(t) = N*, so the model gives
``r N* (1 - N*/K) = h N*``. Dividing by N* (for N* > 0) gives
``N* = K (1 - h/r)``. The sustainable catch is ``Y(h) = h N* = h K (1 - h/r)``,
a downward parabola in h. Setting dY/dh = K(1 - 2h/r) = 0 gives h = r/2, so the
maximum sustainable yield is ``MSY = (r/2) * K * (1/2) = rK/4``. With r = 0.4
and K = 10,000 that is 1,000 tonnes per week at h = 0.2 and N* = 5,000 t.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike

KG_PER_TONNE = 1000
WEEKS_PER_YEAR = 52


def fibonacci(n: int) -> list[int]:
    """First ``n`` Fibonacci numbers starting 1, 1, 2, 3, ... (the old 'stock' baseline)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    seq: list[int] = []
    a, b = 1, 1
    for _ in range(n):
        seq.append(a)
        a, b = b, a + b
    return seq


@dataclass
class SimulationResult:
    """Output of ``FishStock.simulate``: stock has one more entry than harvest."""

    stock: np.ndarray      # tonnes at the start of each week, length weeks + 1
    harvest: np.ndarray    # tonnes landed during each week, length weeks

    @property
    def final_stock(self) -> float:
        return float(self.stock[-1])

    @property
    def total_harvest(self) -> float:
        return float(self.harvest.sum())


@dataclass
class FishStock:
    """Logistic growth with harvesting: N(t+1) = N + r N (1 - N/K) - h N.

    ``closed_weeks`` lists week-of-year indices (0-51) with no fishing; in those
    weeks the harvest term is dropped.
    """

    r: float = 0.4
    K: float = 10_000.0
    N0: float = 4_000.0
    h: float = 0.1
    closed_weeks: frozenset[int] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.r <= 0:
            raise ValueError("growth rate r must be positive")
        if self.K <= 0:
            raise ValueError("carrying capacity K must be positive")
        if self.N0 < 0:
            raise ValueError("initial stock N0 cannot be negative")
        if not 0 <= self.h <= 1:
            raise ValueError("harvest rate h must be between 0 and 1")
        self.closed_weeks = frozenset(int(w) for w in self.closed_weeks)
        if any(not 0 <= w < WEEKS_PER_YEAR for w in self.closed_weeks):
            raise ValueError("closed weeks must be week-of-year indices 0-51")

    def step(self, n: float, week: int = 0) -> tuple[float, float]:
        """Advance one week from stock ``n``; returns (next stock, harvest this week)."""
        # week % 52 converts an absolute week (e.g. 60) into week-of-year (8),
        # so the same closed season repeats every year in multi-year runs.
        harvest = 0.0 if (week % WEEKS_PER_YEAR) in self.closed_weeks else self.h * n
        growth = self.r * n * (1 - n / self.K)   # logistic term: fast when n is small, 0 at n = K
        nxt = n + growth - harvest
        return max(nxt, 0.0), harvest   # a stock cannot go below zero (collapse)

    def simulate(self, weeks: int = WEEKS_PER_YEAR) -> SimulationResult:
        """Run the model for ``weeks`` weeks."""
        if weeks < 1:
            raise ValueError("weeks must be at least 1")
        stock = np.empty(weeks + 1)
        harvest = np.empty(weeks)
        stock[0] = self.N0
        for t in range(weeks):
            stock[t + 1], harvest[t] = self.step(stock[t], t)
        return SimulationResult(stock, harvest)

    # ----- theory -------------------------------------------------------------
    @property
    def msy(self) -> float:
        """Maximum sustainable yield rK/4 (tonnes per week), reached when h = r/2."""
        return self.r * self.K / 4

    def equilibrium_stock(self) -> float:
        """Non-zero steady state K(1 - h/r); 0 if harvesting outpaces growth (h >= r)."""
        return max(self.K * (1 - self.h / self.r), 0.0)

    def equilibrium_yield(self) -> float:
        """Sustainable weekly harvest h * N* at the steady state."""
        return self.h * self.equilibrium_stock()


@dataclass
class PriceModel:
    """Weekly price as a random walk with normal steps, clipped to [low, high].

    Clipping models a market floor/ceiling (below ~9,000 UGX/kg the co-op would
    stop selling to exporters; above ~16,000 buyers substitute other fish).
    """

    start: float = 12_000.0
    low: float = 9_000.0
    high: float = 16_000.0
    step_sd: float = 400.0
    seed: int = 7

    def __post_init__(self) -> None:
        if not self.low < self.high:
            raise ValueError("low bound must be below high bound")
        if not self.low <= self.start <= self.high:
            raise ValueError("start price must lie within the bounds")
        if self.step_sd < 0:
            raise ValueError("step_sd cannot be negative")

    def simulate_paths(self, weeks: int, n_paths: int = 1, seed: int | None = None) -> np.ndarray:
        """Return an (n_paths, weeks) array of prices; week 0 is the start price."""
        if weeks < 1 or n_paths < 1:
            raise ValueError("weeks and n_paths must be positive")
        rng = np.random.default_rng(self.seed if seed is None else seed)
        # Draw every random step up front: one row per path, one column per week.
        # Each week's price = last week's price + a normal shock, then clipped to the bounds.
        steps = rng.normal(0.0, self.step_sd, size=(n_paths, weeks - 1))
        prices = np.empty((n_paths, weeks))
        prices[:, 0] = self.start
        for t in range(1, weeks):  # loop over time so the clip applies at every step
            prices[:, t] = np.clip(prices[:, t - 1] + steps[:, t - 1], self.low, self.high)
        return prices

    def simulate(self, weeks: int) -> np.ndarray:
        """One price path of length ``weeks``."""
        return self.simulate_paths(weeks, 1)[0]


def weekly_revenue(harvest_tonnes: ArrayLike, prices: ArrayLike) -> np.ndarray:
    """Revenue (UGX) = harvest (kg) x price (UGX/kg). Broadcasts over many price paths."""
    return np.asarray(harvest_tonnes, dtype=float) * KG_PER_TONNE * np.asarray(prices, dtype=float)


class RiskAssessor:
    """Classifies revenue risk by the coefficient of variation (CV = sd / mean).

    CV is unit-free, so the same rule works whether revenue is UGX 1 million or
    UGX 1 billion, which a raw variance threshold cannot do. Default bands:

    * CV < 0.10           -> "Low"      (a typical swing is under 10% of income)
    * 0.10 <= CV < 0.25   -> "Moderate"
    * CV >= 0.25          -> "High"     (a one-sigma bad outcome loses a quarter of income)
    """

    def __init__(self, low_cv: float = 0.10, high_cv: float = 0.25) -> None:
        if not 0 < low_cv < high_cv:
            raise ValueError("need 0 < low_cv < high_cv")
        self.low_cv = low_cv
        self.high_cv = high_cv

    @staticmethod
    def summary(revenue: Iterable[float]) -> dict[str, float]:
        """Mean, median, sample variance, sample std and CV using ``statistics``."""
        data = [float(v) for v in revenue]
        if len(data) < 2:
            raise ValueError("need at least two revenue values")
        mean = statistics.mean(data)
        sd = statistics.stdev(data)
        return {"mean": mean, "median": statistics.median(data), "variance": statistics.variance(data),
                "stdev": sd, "cv": sd / mean if mean else float("inf")}

    def classify_cv(self, cv: float) -> str:
        if cv < 0:
            raise ValueError("CV cannot be negative")
        if cv < self.low_cv:
            return "Low"
        return "Moderate" if cv < self.high_cv else "High"

    def classify(self, revenue: Iterable[float]) -> str:
        """Risk class of a revenue sample."""
        return self.classify_cv(self.summary(revenue)["cv"])

    @staticmethod
    def mean_path_cv(revenue_paths: ArrayLike) -> float:
        """Average weekly-revenue CV over many simulated paths, shape (n_paths, weeks).

        One price path can be lucky or unlucky; averaging the CV across paths gives
        the *expected* week-to-week instability of income.
        """
        R = np.atleast_2d(np.asarray(revenue_paths, dtype=float))
        if R.shape[1] < 2:
            raise ValueError("each path needs at least two weeks")
        means = R.mean(axis=1)
        if np.any(means <= 0):
            raise ValueError("CV undefined for paths with no revenue")
        return float(np.mean(R.std(axis=1, ddof=1) / means))

    def classify_paths(self, revenue_paths: ArrayLike) -> str:
        """Risk class based on the expected weekly CV across simulated paths."""
        return self.classify_cv(self.mean_path_cv(revenue_paths))

    @staticmethod
    def value_at_risk(annual_revenue: ArrayLike, level: float = 0.05) -> dict[str, float]:
        """Historical-simulation VaR at ``level``.

        ``quantile`` is the revenue exceeded in (1-level) of simulations; ``var`` is
        the shortfall of that quantile below the mean (how much worse than
        expected a 1-in-20 year is).
        """
        arr = np.asarray(annual_revenue, dtype=float)
        if arr.size == 0:
            raise ValueError("no simulated revenues")
        if not 0 < level < 1:
            raise ValueError("level must be in (0, 1)")
        # The 5 % quantile is the value with 5 % of simulated years below it,
        # i.e. a "1-in-20 bad year".
        q = float(np.quantile(arr, level))
        return {"mean": float(arr.mean()), "quantile": q, "var": float(arr.mean() - q)}

    @staticmethod
    def monte_carlo_annual_revenue(stock: FishStock, prices: PriceModel, n_paths: int = 1000,
                                   weeks: int = WEEKS_PER_YEAR, seed: int | None = None) -> np.ndarray:
        """Annual revenue under ``n_paths`` simulated price paths (harvest is deterministic)."""
        harvest = stock.simulate(weeks).harvest            # shape (weeks,), the same for every path
        paths = prices.simulate_paths(weeks, n_paths, seed)  # shape (n_paths, weeks)
        # Broadcasting multiplies the one harvest vector into every price path;
        # summing along axis=1 (the weeks) gives one annual revenue per path.
        return weekly_revenue(harvest, paths).sum(axis=1)

    def __repr__(self) -> str:
        return f"RiskAssessor(low_cv={self.low_cv}, high_cv={self.high_cv})"
