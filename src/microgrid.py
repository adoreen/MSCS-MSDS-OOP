"""Mini-Project 2: Solar Micro-Grid Dispatch Planner (Kasese health centre).

Each day the energy drawn from solar (x) and battery (y) must satisfy::

    3x + 2y = D1   (daytime load, kWh)
    4x +  y = D2   (critical-equipment load, kWh)

Classes
-------
MicroGrid        holds the coefficient matrix, checks it is well-posed and solves days.
HybridMicroGrid  MicroGrid subclass with a diesel generator (3x3 system).
CostModel        converts dispatched energy into UGX cost.

Helper functions handle validated interactive input, CSV generation/loading and
Monte Carlo sensitivity analysis.

The maths in one place
----------------------
In matrix form the system is ``A @ s = d`` with ``A = [[3, 2], [4, 1]]``,
``s = [x, y]`` (energy per source) and ``d = [D1, D2]`` (demands).

* det(A) = 3*1 - 2*4 = -5. Non-zero, so there is exactly one solution for every d.
* cond(A) = 3 + 2*sqrt(2) ~ 5.83. A relative error in d is amplified at most ~5.8x in s.
* Solving by hand: x = (2*D2 - D1)/5 and y = (4*D1 - 3*D2)/5, so both are
  non-negative only when 0.75*D2 <= D1 <= 2*D2. Outside that band a day is
  physically infeasible and ``dispatch`` switches to non-negative least squares.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import linalg
from scipy.optimize import nnls


class MicroGrid:
    """A linear dispatch model ``A @ s = d`` where ``s`` is energy per source.

    Parameters
    ----------
    coefficients:
        Square coefficient matrix ``A``. Defaults to the 2x2 Kasese system.
    sources:
        Names of the energy sources, one per column of ``A``.
    singular_tol:
        |det(A)| below this is treated as singular (no unique solution).
    """

    DEFAULT_COEFFICIENTS = ((3.0, 2.0), (4.0, 1.0))
    DEFAULT_SOURCES = ("solar", "battery")

    def __init__(
        self,
        coefficients: ArrayLike | None = None,
        sources: Sequence[str] | None = None,
        singular_tol: float = 1e-10,
    ) -> None:
        A = np.asarray(self.DEFAULT_COEFFICIENTS if coefficients is None else coefficients, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"coefficient matrix must be square, got shape {A.shape}")
        sources = tuple(self.DEFAULT_SOURCES if sources is None else sources)
        if len(sources) != A.shape[1]:
            raise ValueError("need exactly one source name per column of the matrix")
        self.A = A
        self.sources = sources
        self.singular_tol = singular_tol

    # ----- well-posedness ---------------------------------------------------
    @property
    def size(self) -> int:
        return self.A.shape[0]

    def determinant(self) -> float:
        """det(A). Zero means the equations are linearly dependent (no unique solution)."""
        return float(np.linalg.det(self.A))

    def condition_number(self) -> float:
        """2-norm condition number: worst-case amplification of relative input error."""
        return float(np.linalg.cond(self.A))

    def is_well_posed(self) -> bool:
        """True when the system has a unique solution (non-singular matrix)."""
        return abs(self.determinant()) > self.singular_tol

    def _require_well_posed(self) -> None:
        if not self.is_well_posed():
            raise np.linalg.LinAlgError(
                f"coefficient matrix is singular (det={self.determinant():.3g}); "
                "the constraints are linearly dependent, so there is no unique dispatch"
            )

    # ----- solving ----------------------------------------------------------
    def _check_demands(self, demands: np.ndarray) -> None:
        if not np.all(np.isfinite(demands)):
            raise ValueError("demands must be finite numbers")
        if np.any(demands < 0):
            raise ValueError("demands cannot be negative")

    def solve_day(self, *demands: float) -> np.ndarray:
        """Solve one day. Pass one demand per constraint, e.g. ``solve_day(d1, d2)``."""
        b = np.asarray(demands, dtype=float)
        if b.size != self.size:
            raise ValueError(f"expected {self.size} demand values, got {b.size}")
        self._check_demands(b)
        self._require_well_posed()
        return linalg.solve(self.A, b)

    def solve_days_loop(self, demands: ArrayLike) -> np.ndarray:
        """Solve many days one at a time in a Python loop. ``demands`` is (n_days, n)."""
        D = self._as_demand_matrix(demands)
        return np.array([self.solve_day(*row) for row in D])

    def solve_days_vectorised(self, demands: ArrayLike) -> np.ndarray:
        """Solve all days in one ``scipy.linalg.solve`` call with an (n, n_days) right-hand side."""
        D = self._as_demand_matrix(demands)
        self._check_demands(D)
        self._require_well_posed()
        # D is (n_days, 2) with one row per day. solve() wants each right-hand side as
        # a *column*, so pass D.T (2 x n_days): A is factorised once and all days are
        # solved together. The result is (2 x n_days), so transpose back to one row per day.
        return linalg.solve(self.A, D.T).T

    def _as_demand_matrix(self, demands: ArrayLike) -> np.ndarray:
        D = np.atleast_2d(np.asarray(demands, dtype=float))
        if D.size == 0:
            raise ValueError("no demand data supplied")
        if D.shape[1] != self.size:
            raise ValueError(f"demands must have {self.size} columns, got {D.shape[1]}")
        return D

    # ----- feasibility ------------------------------------------------------
    @staticmethod
    def infeasible_mask(solutions: ArrayLike, tol: float = 1e-9) -> np.ndarray:
        """Boolean mask of days where any source would have to supply negative energy."""
        S = np.atleast_2d(np.asarray(solutions, dtype=float))
        # A tiny tolerance stops round-off like -1e-15 from being flagged as infeasible.
        # axis=1 checks across the sources of each day (each row).
        return np.any(S < -tol, axis=1)

    def dispatch(self, demands: ArrayLike, dates: Sequence | None = None) -> pd.DataFrame:
        """Solve every day and repair infeasible days with non-negative least squares.

        Strategy: the exact solution is used when it is physically possible. When it
        needs a negative source, we switch to ``scipy.optimize.nnls``, which finds
        the closest non-negative dispatch. That day then cannot meet demand exactly,
        and the size of the shortfall is reported in ``residual_kwh``.
        """
        D = self._as_demand_matrix(demands)
        exact = self.solve_days_vectorised(D)
        bad = self.infeasible_mask(exact)
        final = exact.copy()
        residual = np.zeros(len(D))
        for i in np.flatnonzero(bad):          # indices of the infeasible days only
            # nnls solves min ||A s - d|| subject to s >= 0 and returns (s, ||A s - d||).
            final[i], residual[i] = nnls(self.A, D[i])
        table = pd.DataFrame(D, columns=[f"D{i + 1}" for i in range(self.size)])
        if dates is not None:
            table.insert(0, "date", list(dates))
        for j, src in enumerate(self.sources):
            table[f"{src}_exact"] = exact[:, j]
            table[src] = final[:, j]
        table["feasible"] = ~bad
        table["method"] = np.where(bad, "nnls", "exact")
        table["residual_kwh"] = residual
        return table

    def __repr__(self) -> str:
        return f"{type(self).__name__}(sources={self.sources}, det={self.determinant():.3g}, cond={self.condition_number():.3g})"


class HybridMicroGrid(MicroGrid):
    """Micro-grid with a diesel generator ``z`` and a third constraint.

    Default system (our choice, see the notebook for the justification)::

        3x + 2y + 1z = D1   daytime load
        4x + 1y + 2z = D2   critical-equipment load
        1x + 1y + 1z = D3   total energy drawn from all sources

    Diesel appears with a small weight in the daytime row and a larger weight in
    the critical row, because the generator's job is to back up critical loads.
    """

    DEFAULT_COEFFICIENTS = ((3.0, 2.0, 1.0), (4.0, 1.0, 2.0), (1.0, 1.0, 1.0))
    DEFAULT_SOURCES = ("solar", "battery", "diesel")

    def __init__(self, coefficients: ArrayLike | None = None, sources: Sequence[str] | None = None,
                 singular_tol: float = 1e-10) -> None:
        super().__init__(coefficients, sources, singular_tol)
        if self.size != 3:
            raise ValueError("HybridMicroGrid models exactly three sources")


class CostModel:
    """Energy cost from per-source tariffs (UGX per kWh)."""

    def __init__(self, tariffs: dict[str, float]) -> None:
        if not tariffs:
            raise ValueError("at least one tariff is required")
        if any(v < 0 for v in tariffs.values()):
            raise ValueError("tariffs cannot be negative")
        self.tariffs = dict(tariffs)

    def daily_cost(self, dispatch: pd.DataFrame) -> pd.Series:
        """Cost per day (UGX) = sum over sources of energy x tariff."""
        missing = [s for s in self.tariffs if s not in dispatch]
        if missing:
            raise KeyError(f"dispatch table has no column for {missing}")
        return sum(dispatch[s] * rate for s, rate in self.tariffs.items())

    def monthly_cost(self, dispatch: pd.DataFrame) -> float:
        """Total cost over all days in the table (UGX)."""
        return float(self.daily_cost(dispatch).sum())

    def __repr__(self) -> str:
        return f"CostModel({self.tariffs})"


# ----- input handling -------------------------------------------------------
def parse_demand(text: str) -> float:
    """Convert user text to a non-negative demand in kWh, or raise ``ValueError``."""
    text = text.strip()
    if not text:
        raise ValueError("no value entered")
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"'{text}' is not a number") from None
    if not np.isfinite(value):
        raise ValueError("value must be a finite number")
    if value < 0:
        raise ValueError("demand cannot be negative")
    return value


def prompt_demand(prompt: str, input_fn: Callable[[str], str] = input,
                  print_fn: Callable[[str], None] = print, max_attempts: int | None = None) -> float:
    """Ask for a demand until a valid value is entered (re-prompts on bad input).

    ``input_fn``/``print_fn`` are injectable so the loop can be unit-tested and run
    non-interactively in the notebook. ``max_attempts=None`` means keep asking.
    """
    attempts = 0
    while max_attempts is None or attempts < max_attempts:
        attempts += 1
        try:
            return parse_demand(input_fn(prompt))
        except ValueError as err:
            print_fn(f"  Invalid input: {err}. Please try again.")
    raise ValueError(f"no valid input after {max_attempts} attempts")


# ----- data generation & loading -------------------------------------------
def generate_demand_csv(path: str | Path, days: int = 30, seed: int = 42,
                        start: str = "2026-09-01") -> pd.DataFrame:
    """Write a synthetic (illustrative) 30-day demand file and return it.

    Assumptions: the health centre runs outpatient clinics Monday-Friday, so the
    daytime load D1 is higher on weekdays (~105 kWh) than at weekends (~80 kWh).
    The critical-equipment load D2 (fridges for vaccines, oxygen concentrators,
    lighting for the maternity ward) runs every day at ~58 kWh with only small
    variation. Both get Gaussian noise. Values are rounded to 0.1 kWh, like a
    meter reading.
    """
    if days < 1:
        raise ValueError("days must be at least 1")
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days, freq="D")
    weekday = dates.dayofweek.to_numpy() < 5          # Monday=0 ... Friday=4 -> True
    # weekly pattern (weekday vs weekend level) + random noise
    d1 = np.where(weekday, 105.0, 80.0) + rng.normal(0, 8.0, days)
    d2 = 58.0 + 3.0 * weekday + rng.normal(0, 4.0, days)   # True counts as 1, False as 0
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"),
                       "D1": np.round(np.clip(d1, 0, None), 1),
                       "D2": np.round(np.clip(d2, 0, None), 1)})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_demand_csv(path: str | Path) -> pd.DataFrame:
    """Load a demand CSV with columns ``date, D1, D2`` and validate it."""
    df = pd.read_csv(path)
    required = {"D1", "D2"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {sorted(required)}")
    if df.empty:
        raise ValueError("CSV contains no rows")
    if df[["D1", "D2"]].isna().any().any() or (df[["D1", "D2"]] < 0).any().any():
        raise ValueError("CSV contains missing or negative demand values")
    return df


# ----- statistics & sensitivity --------------------------------------------
def usage_statistics(values: Sequence[float]) -> dict[str, float]:
    """Mean, sample variance, sample std and coefficient of variation via ``statistics``."""
    data = [float(v) for v in values]
    if len(data) < 2:
        raise ValueError("need at least two values for a sample variance")
    mean = statistics.mean(data)
    sd = statistics.stdev(data)
    return {"mean": mean, "variance": statistics.variance(data), "stdev": sd,
            "cv": sd / mean if mean else float("nan")}


def monte_carlo_sensitivity(grid: MicroGrid, demands: Sequence[float], rel: float = 0.05,
                            n_draws: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Perturb every demand independently by up to +/-``rel`` (uniform) and re-solve.

    Returns one row per draw with the perturbed demands and the resulting dispatch.
    """
    if n_draws < 1:
        raise ValueError("n_draws must be positive")
    rng = np.random.default_rng(seed)
    base = np.asarray(demands, dtype=float)
    # each draw multiplies every demand by its own factor in [0.95, 1.05]
    factors = rng.uniform(1 - rel, 1 + rel, size=(n_draws, base.size))
    D = base * factors                       # broadcasting: (n_draws, n) * (n,)
    S = grid.solve_days_vectorised(D)        # 1,000 systems in one call
    out = pd.DataFrame(D, columns=[f"D{i + 1}" for i in range(base.size)])
    for j, src in enumerate(grid.sources):
        out[src] = S[:, j]
    return out
