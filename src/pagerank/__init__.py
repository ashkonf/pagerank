import math
from typing import Mapping, Sequence, cast

import numpy
import pandas

# Generalized matrix operations:


def __extract_nodes(matrix: pandas.DataFrame) -> set[str]:
    """Extract all unique node names from a transition matrix.

    Args:
        matrix: A pandas DataFrame representing the transition matrix.

    Returns:
        A set containing all unique node names from both rows and columns.
    """
    nodes = set()
    for col_key in matrix:
        nodes.add(col_key)
    for row_key in matrix.T:
        nodes.add(row_key)
    return nodes


def __make_square(
    matrix: pandas.DataFrame, keys: set[str], default: float = 0.0
) -> pandas.DataFrame:
    """Make a matrix square by adding missing rows and columns.

    Args:
        matrix: The input pandas DataFrame to make square.
        keys: Set of all keys that should be present as both rows and columns.
        default: Default value to use for missing entries.

    Returns:
        A square pandas DataFrame with all keys as both rows and columns.
    """
    matrix = matrix.copy()

    def insert_missing_columns(matrix: pandas.DataFrame) -> pandas.DataFrame:
        for key in keys:
            if key not in matrix:
                matrix[key] = pandas.Series(default, index=matrix.index)
        return matrix

    matrix = insert_missing_columns(matrix)
    matrix = insert_missing_columns(matrix.T).T

    return matrix.fillna(default)


def __ensure_rows_positive(matrix: pandas.DataFrame) -> pandas.DataFrame:
    """Ensure all rows have positive sums by replacing zero-sum rows with uniform distribution.

    Args:
        matrix: The input pandas DataFrame to process.

    Returns:
        A pandas DataFrame where all rows have positive sums.
    """
    matrix = matrix.T
    for col_key in matrix:
        if matrix[col_key].sum() == 0.0:
            matrix[col_key] = pandas.Series(
                numpy.ones(len(matrix[col_key])), index=matrix.index
            )
    return matrix.T


def __normalize_rows(matrix: pandas.DataFrame) -> pandas.DataFrame:
    """Normalize each row to sum to 1.

    Args:
        matrix: The input pandas DataFrame to normalize.

    Returns:
        A pandas DataFrame with each row normalized to sum to 1.
    """
    return matrix.div(matrix.sum(axis=1), axis=0)


def __euclidean_norm(series: pandas.Series) -> float:
    """Calculate the Euclidean norm of a pandas Series.

    Args:
        series: The input pandas Series.

    Returns:
        The Euclidean norm as a float.
    """
    dot_product = float(series.dot(series))
    return math.sqrt(dot_product)


# PageRank specific functionality:


def __start_state(nodes: set[str]) -> pandas.Series:
    """Create initial uniform probability distribution over nodes.

    Args:
        nodes: Set of node names.

    Returns:
        A pandas Series with uniform probability distribution.

    Raises:
        ValueError: If no nodes are provided.
    """
    if len(nodes) == 0:
        raise ValueError("There must be at least one node.")
    start_prob = 1.0 / float(len(nodes))
    return pandas.Series(dict.fromkeys(nodes, start_prob))


def __integrate_random_surfer(
    nodes: set[str], transition_probabilities: pandas.DataFrame, rsp: float
) -> pandas.DataFrame:
    """Integrate random surfer probability into transition matrix.

    Args:
        nodes: Set of all node names.
        transition_probabilities: The transition probability matrix.
        rsp: Random surfer probability (damping factor).

    Returns:
        Modified transition matrix with random surfer probability integrated.
    """
    alpha = 1.0 / float(len(nodes)) * rsp
    return transition_probabilities.copy().multiply(1.0 - rsp) + alpha


def power_iteration(
    transition_weights: Mapping[str, Mapping[str, float | int]]
    | Sequence[Sequence[float | int]],
    rsp: float = 0.15,
    epsilon: float = 0.00001,
    max_iterations: int = 1000,
) -> pandas.Series:
    """Apply PageRank algorithm using power iteration to find steady-state probabilities.

    This function applies the PageRank algorithm to a provided graph to determine
    the steady probabilities with which a random walk through the graph will end up
    at each node. It uses power iteration, an algorithm that iteratively refines
    the steady state probabilities until convergence.

    Args:
        transition_weights: Sparse representation of the graph as nested dicts or lists.
            Keys correspond to node names and values to weights. Elements need not be
            probabilities (rows need not be normalized).
        rsp: Random surfer probability controlling the chance of jumping to any node.
            Also known as the damping factor (1 - rsp is the damping factor).
        epsilon: Threshold of convergence; iteration stops when successive approximations
            are closer than this value.
        max_iterations: Maximum number of iterations before termination even without convergence.

    Returns:
        A pandas Series whose keys are node names and whose values are the corresponding
        steady state probabilities. This series can be treated as a dict.

    Example:
        >>> import logging
        >>> logging.basicConfig(level=logging.INFO)
        >>> graph = {
        ...     "A": {"B": 1, "C": 1},
        ...     "B": {"C": 1},
        ...     "C": {"A": 1},
        ... }
        >>> scores = power_iteration(graph)
        >>> logging.info(scores)
    """
    transition_weights_df = pandas.DataFrame(transition_weights)
    nodes = __extract_nodes(transition_weights_df)
    transition_weights_df = __make_square(transition_weights_df, nodes, default=0.0)
    transition_weights_df = __ensure_rows_positive(transition_weights_df)

    state = __start_state(nodes)
    transition_probabilities = __normalize_rows(transition_weights_df)
    transition_probabilities = __integrate_random_surfer(
        nodes, transition_probabilities, rsp
    )

    for _iteration in range(max_iterations):
        old_state = state.copy()
        state = cast(
            pandas.Series,
            state.dot(transition_probabilities),  # type: ignore[reportArgumentType]
        )
        delta = state - old_state
        if __euclidean_norm(delta) < epsilon:
            break

    return cast(pandas.Series, state)
