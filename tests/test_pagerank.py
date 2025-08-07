import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import pagerank


def test_extract_nodes() -> None:
    matrix = pd.DataFrame({"A": {"B": 1.0}, "C": {"B": 2.0}})
    nodes = pagerank.__extract_nodes(matrix)
    assert nodes == {"A", "B", "C"}


def test_make_square() -> None:
    matrix = pd.DataFrame({"A": {"A": 1.0}, "B": {"A": 2.0}})
    keys = {"A", "B", "C"}
    square = pagerank.__make_square(matrix, keys, default=0.0)
    assert set(square.index) == keys
    assert set(square.columns) == keys
    assert square.loc["C", "A"] == 0.0


def test_ensure_rows_positive() -> None:
    matrix = pd.DataFrame({"A": {"A": 0.0, "B": 1.0}, "B": {"A": 0.0, "B": 0.0}})
    result = pagerank.__ensure_rows_positive(matrix)
    assert (result.loc["A"] == pd.Series({"A": 1.0, "B": 1.0})).all()
    assert (result.loc["B"] == pd.Series({"A": 1.0, "B": 0.0})).all()


def test_normalize_rows() -> None:
    matrix = pd.DataFrame({"A": {"A": 1.0, "B": 1.0}, "B": {"A": 1.0, "B": 1.0}})
    normalized = pagerank.__normalize_rows(matrix)
    assert normalized.loc["A", "A"] == pytest.approx(0.5)
    assert normalized.loc["A", "B"] == pytest.approx(0.5)


def test_euclidean_norm() -> None:
    series = pd.Series([3.0, 4.0])
    assert pagerank.__euclidean_norm(series) == pytest.approx(5.0)


def test_start_state() -> None:
    nodes = {"A", "B"}
    state = pagerank.__start_state(nodes)
    assert state.sum() == pytest.approx(1.0)
    assert all(state == 0.5)


def test_start_state_empty() -> None:
    with pytest.raises(ValueError):
        pagerank.__start_state(set())


def test_integrate_random_surfer() -> None:
    nodes = {"A", "B"}
    transition = pd.DataFrame({"A": {"A": 1.0, "B": 0.0}, "B": {"A": 0.0, "B": 1.0}})
    integrated = pagerank.__integrate_random_surfer(nodes, transition, 0.15)
    assert integrated.loc["A", "A"] == pytest.approx(0.925)
    assert integrated.loc["A", "B"] == pytest.approx(0.075)


def test_power_iteration() -> None:
    weights = {"A": {"B": 1.0}, "B": {"A": 1.0}}
    result = pagerank.power_iteration(
        weights, rsp=0.0, epsilon=1e-6, max_iterations=100
    )
    assert result["A"] == pytest.approx(0.5)
    assert result["B"] == pytest.approx(0.5)
