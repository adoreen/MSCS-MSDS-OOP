"""Tests for Mini-Project 4 (rainfall & crop suitability)."""

import numpy as np
import pytest

from src.rainfall import (CropRule, Region, SimilarityAnalyser, climatology, cosine_similarity,
                          scipy_cosine_similarity, suitability_table)

KAMPALA = [120, 140, 180, 200, 220, 180, 90, 70, 60, 100, 110, 130]
MBARARA = [70, 85, 120, 140, 90, 25, 20, 55, 100, 125, 120, 90]


def test_region_summary_methods():
    r = Region("Kampala", KAMPALA)
    assert r.annual_total() == 1600
    assert r.mean() == pytest.approx(1600 / 12)
    assert r.wettest_month() == "May" and r.driest_month() == "Sep"
    assert r.coefficient_of_variation() == pytest.approx(np.std(KAMPALA, ddof=1) / np.mean(KAMPALA))


@pytest.mark.parametrize("bad", [[1] * 11, [1] * 11 + [-5], []])
def test_region_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        Region("Bad", bad)


def test_cosine_similarity_matches_scipy_and_ignores_scale():
    a, b = np.array(KAMPALA), np.array(MBARARA)
    assert cosine_similarity(a, b) == pytest.approx(scipy_cosine_similarity(a, b))
    assert cosine_similarity(a, 3 * a) == pytest.approx(1.0)   # 3x wetter, still "identical"
    with pytest.raises(ValueError):
        cosine_similarity(a, np.zeros(12))


def test_peak_detection_handles_wraparound():
    # single season peaking in January: invisible at the array edge without wrap-around
    jan_peak = Region("Jan", [200, 150, 80, 40, 20, 10, 10, 10, 20, 40, 80, 150])
    assert jan_peak.detect_peaks() == [0]
    assert jan_peak.modality() == "unimodal"
    assert Region("Mbarara", MBARARA).modality() == "bimodal"
    assert Region("Flat", [50] * 12).modality() == "no clear season"


def test_crop_rule_classification():
    maize = CropRule.from_seasonal_need("Maize", 500, 800, 4, source="test")
    assert (maize.min_mm, maize.max_mm) == (125, 200)
    assert maize.classify(150) == "Good for maize"
    assert maize.classify(50) == "Drought risk"
    assert maize.classify(250) == "Waterlogging risk"
    assert [maize.score(v) for v in (50, 150, 250)] == [-1, 0, 1]
    with pytest.raises(ValueError):
        CropRule("Bad", 100, 50)


def test_similarity_matrices_are_symmetric():
    sa = SimilarityAnalyser([Region("Kampala", KAMPALA), Region("Mbarara", MBARARA)])
    for m in (sa.cosine(), sa.pearson(), sa.euclidean()):
        np.testing.assert_allclose(m.values, m.values.T)
    assert sa.euclidean().loc["Kampala", "Kampala"] == 0
    with pytest.raises(ValueError):
        SimilarityAnalyser([Region("Kampala", KAMPALA)])


def test_suitability_table_shape():
    crops = [CropRule("A", 50, 150), CropRule("B", 100, 200)]
    table = suitability_table([Region("Kampala", KAMPALA)], crops)
    assert len(table) == 24 and set(table["score"]) <= {-1, 0, 1}


def test_climatology_from_long_table():
    import pandas as pd
    rows = [{"region": "X", "year": y, "month": m, "rain_mm": float(m + (y - 2020))}
            for y in (2020, 2022) for m in range(1, 13)]
    region = climatology(pd.DataFrame(rows), "X")
    np.testing.assert_allclose(region.rainfall, np.arange(1, 13) + 1.0)   # mean of +0 and +2
    with pytest.raises(ValueError):
        climatology(pd.DataFrame(rows), "Missing")
