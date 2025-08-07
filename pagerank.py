from __future__ import annotations

import math
from typing import Any, Dict, Set, cast

import numpy as np
import pandas as pd


# Generalized matrix operations:


def __extract_nodes(matrix: pd.DataFrame) -> Set[str]:
    nodes: Set[str] = set()
    for col_key in matrix:
        nodes.add(col_key)
    for row_key in matrix.T:
        nodes.add(row_key)
    return nodes


def __make_square(
    matrix: pd.DataFrame, keys: Set[str], default: float = 0.0
) -> pd.DataFrame:
    result = matrix.copy()

    def insert_missing_columns(m: pd.DataFrame) -> pd.DataFrame:
        for key in keys:
            if key not in m:
                m[key] = pd.Series(default, index=m.index)
        return m

    result = insert_missing_columns(result)
    result = insert_missing_columns(result.T).T

    return result.fillna(default)


def __ensure_rows_positive(matrix: pd.DataFrame) -> pd.DataFrame:
    result = matrix.T
    for col_key in result:
        if result[col_key].sum() == 0.0:
            result[col_key] = pd.Series(
                np.ones(len(result[col_key])), index=result.index
            )
    return result.T


def __normalize_rows(matrix: pd.DataFrame) -> pd.DataFrame:
    return matrix.div(matrix.sum(axis=1), axis=0)


def __euclidean_norm(series: pd.Series) -> float:
    return float(math.sqrt(series.dot(series)))


# PageRank specific functionality:


def __start_state(nodes: Set[str]) -> pd.Series:
    if len(nodes) == 0:
        raise ValueError("There must be at least one node.")
    start_prob: float = 1.0 / float(len(nodes))
    return pd.Series({node: start_prob for node in nodes})


def __integrate_random_surfer(
    nodes: Set[str], transition_probabilities: pd.DataFrame, rsp: float
) -> pd.DataFrame:
    alpha: float = 1.0 / float(len(nodes)) * rsp
    return transition_probabilities.copy().multiply(1.0 - rsp) + alpha


def power_iteration(
    transition_weights: Dict[str, Dict[str, float]] | pd.DataFrame,
    rsp: float = 0.15,
    epsilon: float = 0.00001,
    max_iterations: int = 1000,
) -> pd.Series:
    # Clerical work:
    transition_weights_df = pd.DataFrame(transition_weights)
    nodes = __extract_nodes(transition_weights_df)
    transition_weights_df = __make_square(transition_weights_df, nodes, default=0.0)
    transition_weights_df = __ensure_rows_positive(transition_weights_df)

    # Setup:
    state = __start_state(nodes)
    transition_probabilities = __normalize_rows(transition_weights_df)
    transition_probabilities = __integrate_random_surfer(
        nodes, transition_probabilities, rsp
    )

    # Power iteration:
    for _ in range(max_iterations):
        old_state = state.copy()
        state = cast(pd.Series, state.dot(cast(Any, transition_probabilities)))
        delta = state - old_state
        if __euclidean_norm(delta) < epsilon:
            break

    return state
