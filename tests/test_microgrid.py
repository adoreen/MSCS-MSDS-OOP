"""Tests for Mini-Project 2 (micro-grid dispatch)."""

import numpy as np
import pandas as pd
import pytest

from src.microgrid import (CostModel, HybridMicroGrid, MicroGrid, generate_demand_csv,
                           load_demand_csv, monte_carlo_sensitivity, parse_demand,
                           prompt_demand, usage_statistics)


@pytest.fixture
def grid() -> MicroGrid:
    return MicroGrid()


def test_determinant_and_condition_number(grid):
    assert grid.determinant() == pytest.approx(-5)        # 3*1 - 2*4
    assert grid.condition_number() == pytest.approx(5.8284, rel=1e-4)  # = 3 + 2*sqrt(2)
    assert grid.is_well_posed()


def test_solve_day_matches_hand_solution(grid):
    # x = (2*D2 - D1)/5, y = (4*D1 - 3*D2)/5
    np.testing.assert_allclose(grid.solve_day(100, 60), [4, 44])


def test_loop_and_vectorised_agree(grid):
    rng = np.random.default_rng(0)
    D = rng.uniform(50, 120, size=(30, 2))
    np.testing.assert_allclose(grid.solve_days_loop(D), grid.solve_days_vectorised(D))


def test_negative_or_wrong_size_demand_rejected(grid):
    with pytest.raises(ValueError):
        grid.solve_day(-1, 50)
    with pytest.raises(ValueError):
        grid.solve_day(10)
    with pytest.raises(ValueError):
        grid.solve_days_vectorised(np.empty((0, 2)))


def test_infeasible_day_repaired_with_nnls(grid):
    # D1 > 2*D2 forces negative solar in the exact solution
    table = grid.dispatch([[100, 60], [130, 50]])
    assert table["feasible"].tolist() == [True, False]
    assert (table[["solar", "battery"]] >= 0).all().all()
    assert table.loc[1, "residual_kwh"] > 0


@pytest.mark.parametrize("text, expected", [("12.5", 12.5), (" 0 ", 0.0)])
def test_parse_demand_valid(text, expected):
    assert parse_demand(text) == expected


@pytest.mark.parametrize("text", ["", "   ", "abc", "-3", "nan"])
def test_parse_demand_invalid(text):
    with pytest.raises(ValueError):
        parse_demand(text)


def test_prompt_reprompts_until_valid():
    answers = iter(["", "ten", "-5", "42"])
    messages = []
    value = prompt_demand("D1: ", input_fn=lambda _: next(answers), print_fn=messages.append)
    assert value == 42 and len(messages) == 3


def test_csv_round_trip_is_reproducible(tmp_path):
    a = generate_demand_csv(tmp_path / "a.csv", seed=1)
    b = generate_demand_csv(tmp_path / "b.csv", seed=1)
    pd.testing.assert_frame_equal(a, b)
    loaded = load_demand_csv(tmp_path / "a.csv")
    assert len(loaded) == 30 and (loaded[["D1", "D2"]] >= 0).all().all()


def test_cost_model_and_statistics():
    table = pd.DataFrame({"solar": [10.0, 20.0], "battery": [1.0, 2.0]})
    cost = CostModel({"solar": 150, "battery": 450})
    assert cost.daily_cost(table).tolist() == [1950, 3900]
    assert cost.monthly_cost(table) == 5850
    assert usage_statistics([1, 2, 3])["variance"] == pytest.approx(1)
    with pytest.raises(ValueError):
        usage_statistics([5])


def test_hybrid_grid_and_dependent_constraint():
    hybrid = HybridMicroGrid()
    sol = hybrid.solve_day(60, 70, 25)
    np.testing.assert_allclose(hybrid.A @ sol, [60, 70, 25])
    dependent = HybridMicroGrid([[3, 2, 1], [4, 1, 2], [7, 3, 3]])   # row3 = row1 + row2
    assert not dependent.is_well_posed()
    with pytest.raises(np.linalg.LinAlgError):
        dependent.solve_day(60, 70, 130)


def test_monte_carlo_sensitivity_is_seeded(grid):
    a = monte_carlo_sensitivity(grid, [100, 60], n_draws=50, seed=3)
    b = monte_carlo_sensitivity(grid, [100, 60], n_draws=50, seed=3)
    pd.testing.assert_frame_equal(a, b)
    assert ((a["D1"] >= 95) & (a["D1"] <= 105)).all()
