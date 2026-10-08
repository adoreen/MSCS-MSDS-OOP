"""Mini-Project 4: Rainfall Pattern & Crop Suitability Analyser.

Classes
-------
Region              one region's 12 monthly rainfall totals (mm) plus summary methods.
CropRule            a crop's suitable monthly rainfall band and a month classifier.
SimilarityAnalyser  pairwise cosine / Pearson / Euclidean matrices between regions.

Plus two helpers for the real-data extension: ``fetch_nasa_power_monthly``
(download) and ``climatology`` (average each calendar month into a ``Region``).

The three comparison measures, for rainfall vectors a and b (12 months each)
--------------------------------------------------------------------------
* cosine similarity  = a.b / (|a||b|)   the angle between them; ignores overall wetness
* Pearson r          = cosine of (a - mean a) and (b - mean b); compares timing only
* Euclidean distance = |a - b|           absolute mm difference; timing and amount
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.signal import find_peaks
from scipy.spatial.distance import cosine as scipy_cosine_distance

MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


class Region:
    """Monthly rainfall climatology (Jan-Dec, mm) for one region."""

    def __init__(self, name: str, rainfall: ArrayLike) -> None:
        arr = np.asarray(rainfall, dtype=float)
        if not name or not str(name).strip():
            raise ValueError("region name must not be empty")
        if arr.shape != (12,):
            raise ValueError(f"rainfall must have exactly 12 monthly values, got shape {arr.shape}")
        if not np.all(np.isfinite(arr)) or np.any(arr < 0):
            raise ValueError("rainfall must be finite and non-negative")
        self.name = str(name).strip()
        self.rainfall = arr

    def __repr__(self) -> str:
        return f"Region({self.name!r}, annual={self.annual_total():.0f} mm)"

    def __len__(self) -> int:
        return 12

    def annual_total(self) -> float:
        return float(self.rainfall.sum())

    def mean(self) -> float:
        return float(self.rainfall.mean())

    def wettest_month(self) -> str:
        return MONTHS[int(np.argmax(self.rainfall))]

    def driest_month(self) -> str:
        return MONTHS[int(np.argmin(self.rainfall))]

    def coefficient_of_variation(self) -> float:
        """Sample std / mean. High CV = rain concentrated in a few months."""
        mean = statistics.mean(self.rainfall.tolist())
        if mean == 0:
            raise ValueError("CV undefined for a region with no rain")
        return statistics.stdev(self.rainfall.tolist()) / mean

    def detect_peaks(self, relative_prominence: float = 0.25) -> list[int]:
        """Month indices (0-11) of rainy-season peaks found by ``scipy.signal.find_peaks``.

        Two adjustments are needed for monthly climate data:

        * **Wrap-around.** December is next to January, so a season peaking in
          December/January would be missed at the array edge. We tile the year
          three times, detect peaks, and keep only those in the middle copy.
        * **Prominence.** A peak only counts as a separate season if it rises at
          least ``relative_prominence`` x (annual max - annual min) above the
          troughs around it. Expressing this relative to each region's own range
          keeps the rule scale-free, like the CV.
        """
        if not 0 <= relative_prominence < 1:
            raise ValueError("relative_prominence must be in [0, 1)")
        span = float(self.rainfall.max() - self.rainfall.min())
        if span == 0:
            return []  # perfectly flat rainfall has no seasons
        # [Jan..Dec, Jan..Dec, Jan..Dec]: the middle copy (indices 12-23) has real
        # neighbours on both sides, so December sees January and vice versa.
        tiled = np.tile(self.rainfall, 3)
        peaks, _ = find_peaks(tiled, prominence=relative_prominence * span)
        # keep peaks in the middle copy and convert back to month numbers 0-11
        return sorted(int(p - 12) for p in peaks if 12 <= p < 24)

    def peak_prominences(self) -> dict[str, float]:
        """Every local peak and its prominence as a fraction of the annual range."""
        span = float(self.rainfall.max() - self.rainfall.min()) or 1.0
        tiled = np.tile(self.rainfall, 3)
        peaks, props = find_peaks(tiled, prominence=0)
        return {MONTHS[p - 12]: float(prom / span)
                for p, prom in zip(peaks, props["prominences"]) if 12 <= p < 24}

    def modality(self, relative_prominence: float = 0.25) -> str:
        """'unimodal', 'bimodal', 'multimodal' or 'no clear season'."""
        n = len(self.detect_peaks(relative_prominence))
        # dict lookup with a default: 3 or more peaks fall through to "multimodal"
        return {0: "no clear season", 1: "unimodal", 2: "bimodal"}.get(n, "multimodal")


@dataclass(frozen=True)
class CropRule:
    """Suitable monthly rainfall band for one crop.

    ``min_mm``/``max_mm`` are monthly figures. ``from_seasonal_need`` derives
    them from the total water a crop needs over its growing period, which is how
    agronomic sources usually report it.
    """

    crop: str
    min_mm: float
    max_mm: float
    source: str = ""

    def __post_init__(self) -> None:
        if not self.crop:
            raise ValueError("crop name required")
        if self.min_mm < 0 or self.max_mm <= self.min_mm:
            raise ValueError("need 0 <= min_mm < max_mm")

    @classmethod
    def from_seasonal_need(cls, crop: str, season_min_mm: float, season_max_mm: float,
                           season_months: float, source: str = "") -> "CropRule":
        """Spread a growing-season water requirement evenly over its months."""
        if season_months <= 0:
            raise ValueError("season_months must be positive")
        return cls(crop, season_min_mm / season_months, season_max_mm / season_months, source)

    def classify(self, rainfall_mm: float) -> str:
        """'Good for <crop>', 'Drought risk' or 'Waterlogging risk' for one month."""
        if rainfall_mm < 0:
            raise ValueError("rainfall cannot be negative")
        if rainfall_mm < self.min_mm:
            return "Drought risk"
        if rainfall_mm > self.max_mm:
            return "Waterlogging risk"
        return f"Good for {self.crop.lower()}"

    def score(self, rainfall_mm: float) -> int:
        """Numeric code for heatmaps: -1 too dry, 0 suitable, +1 too wet."""
        if rainfall_mm < self.min_mm:
            return -1
        return 1 if rainfall_mm > self.max_mm else 0


def suitability_table(regions: Sequence[Region], crops: Sequence[CropRule]) -> pd.DataFrame:
    """Long table with one row per region x crop x month."""
    rows = [{"region": r.name, "crop": c.crop, "month": MONTHS[m], "rain_mm": r.rainfall[m],
             "status": c.classify(r.rainfall[m]), "score": c.score(r.rainfall[m])}
            for r in regions for c in crops for m in range(12)]
    return pd.DataFrame(rows)


def cosine_similarity(a: ArrayLike, b: ArrayLike) -> float:
    """cos(theta) = (a . b) / (|a| |b|): the angle between two rainfall profiles."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("vectors must be non-empty and the same length")
    na, nb = np.linalg.norm(a), np.linalg.norm(b)   # vector lengths sqrt(sum a_i^2)
    if na == 0 or nb == 0:
        raise ValueError("cosine similarity is undefined for a zero vector")   # would divide by 0
    return float(np.dot(a, b) / (na * nb))


def scipy_cosine_similarity(a: ArrayLike, b: ArrayLike) -> float:
    """Reference value: SciPy returns cosine *distance*, so similarity = 1 - distance."""
    return float(1 - scipy_cosine_distance(a, b))


NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/monthly/point"


def fetch_nasa_power_monthly(name: str, lat: float, lon: float, start: int, end: int) -> pd.DataFrame:
    """Download monthly precipitation totals (mm, PRECTOTCORR_SUM) from NASA POWER.

    Returns a long table with columns region, year, month (1-12), rain_mm. The API
    also returns a month "13" (annual total), which we drop.
    """
    import json
    import urllib.request
    from urllib.parse import urlencode

    if start > end:
        raise ValueError("start year must not be after end year")
    query = urlencode({"parameters": "PRECTOTCORR_SUM", "community": "AG", "latitude": lat,
                       "longitude": lon, "start": start, "end": end, "format": "JSON"})
    with urllib.request.urlopen(f"{NASA_POWER_URL}?{query}", timeout=120) as resp:
        payload = json.load(resp)
    series = payload["properties"]["parameter"]["PRECTOTCORR_SUM"]
    rows = [{"region": name, "year": int(k[:4]), "month": int(k[4:]), "rain_mm": float(v)}
            for k, v in series.items() if int(k[4:]) <= 12 and v >= 0]  # -999 = missing
    return pd.DataFrame(rows)


def climatology(long_table: pd.DataFrame, region: str) -> Region:
    """Average each calendar month over all years to build a ``Region``."""
    sub = long_table[long_table["region"] == region]
    if sub.empty:
        raise ValueError(f"no data for region {region!r}")
    means = sub.groupby("month")["rain_mm"].mean().reindex(range(1, 13))
    if means.isna().any():
        raise ValueError(f"{region} is missing at least one calendar month")
    return Region(region, means.to_numpy())


class SimilarityAnalyser:
    """Pairwise similarity/distance matrices for a set of regions."""

    def __init__(self, regions: Sequence[Region]) -> None:
        if len(regions) < 2:
            raise ValueError("need at least two regions to compare")
        names = [r.name for r in regions]
        if len(set(names)) != len(names):
            raise ValueError("region names must be unique")
        self.regions = list(regions)

    def _matrix(self, func) -> pd.DataFrame:
        """Apply ``func(a, b)`` to every pair of regions. The three public methods
        differ only in the function they pass, so the looping code is written once."""
        names = [r.name for r in self.regions]
        data = [[func(a.rainfall, b.rainfall) for b in self.regions] for a in self.regions]
        return pd.DataFrame(data, index=names, columns=names)

    def cosine(self) -> pd.DataFrame:
        return self._matrix(cosine_similarity)

    def pearson(self) -> pd.DataFrame:
        return self._matrix(lambda a, b: float(np.corrcoef(a, b)[0, 1]))

    def euclidean(self) -> pd.DataFrame:
        return self._matrix(lambda a, b: float(np.linalg.norm(a - b)))
