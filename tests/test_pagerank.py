import os
import sys
import pytest

# Ensure the project root is on the path so that 'pagerank' can be imported
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from pagerank import power_iteration


def test_two_node_cycle_uniform_distribution():
    graph = {'a': {'b': 1}, 'b': {'a': 1}}
    result = power_iteration(graph)
    assert result['a'] == pytest.approx(0.5)
    assert result['b'] == pytest.approx(0.5)
    assert result.sum() == pytest.approx(1.0)


def test_chain_with_sink_distribution():
    graph = {'A': {'B': 1}, 'B': {'C': 1}, 'C': {}}
    result = power_iteration(graph)
    expected = {
        'A': 0.4744121715076071,
        'B': 0.3411710465652372,
        'C': 0.18441678192715547,
    }
    for node, value in expected.items():
        assert result[node] == pytest.approx(value, abs=1e-6)
    assert result.sum() == pytest.approx(1.0)


def test_empty_graph_raises_value_error():
    with pytest.raises(ValueError):
        power_iteration({})


def test_random_surfer_one_uniform_distribution():
    graph = {'A': {'B': 1}, 'B': {'C': 1}, 'C': {'A': 1}}
    result = power_iteration(graph, rsp=1.0)
    for node in graph:
        assert result[node] == pytest.approx(1 / 3, abs=1e-6)
    assert result.sum() == pytest.approx(1.0)


def test_weighted_graph_distribution():
    graph = {
        'A': {'B': 2, 'C': 1},
        'B': {'C': 1},
        'C': {'A': 1, 'B': 1},
    }
    result = power_iteration(graph)
    expected = {
        'A': 0.355920,
        'B': 0.227183,
        'C': 0.416897,
    }
    for node, value in expected.items():
        assert result[node] == pytest.approx(value, abs=1e-6)
    assert result.sum() == pytest.approx(1.0)


def test_chain_without_random_surfer_distribution():
    graph = {'A': {'B': 1}, 'B': {'C': 1}, 'C': {'C': 1}}
    result = power_iteration(graph, rsp=0.0)
    expected = {
        'A': 0.428572,
        'B': 0.285714,
        'C': 0.285714,
    }
    for node, value in expected.items():
        assert result[node] == pytest.approx(value, abs=1e-6)
    assert result.sum() == pytest.approx(1.0)
