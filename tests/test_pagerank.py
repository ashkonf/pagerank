"""Tests for the pagerank module."""

import pytest
import pandas as pd
import numpy as np
from pagerank import power_iteration


class TestPowerIteration:
    """Test cases for the power_iteration function."""
    
    def test_simple_graph(self):
        """Test PageRank on a simple 3-node graph."""
        graph = {
            "A": {"B": 1, "C": 1},
            "B": {"C": 1},
            "C": {"A": 1},
        }
        
        scores = power_iteration(graph)
        
        assert isinstance(scores, pd.Series)
        assert len(scores) == 3
        assert set(scores.index) == {"A", "B", "C"}
        assert abs(scores.sum() - 1.0) < 1e-10
        assert all(score > 0 for score in scores.values)
    
    def test_single_node(self):
        """Test PageRank on a single node graph."""
        graph = {"A": {}}
        
        scores = power_iteration(graph)
        
        assert len(scores) == 1
        assert scores["A"] == 1.0
    
    def test_disconnected_nodes(self):
        """Test PageRank on a graph with disconnected nodes."""
        graph = {
            "A": {"B": 1},
            "B": {},
            "C": {}
        }
        
        scores = power_iteration(graph)
        
        assert len(scores) == 3
        assert abs(scores.sum() - 1.0) < 1e-10
        assert all(score > 0 for score in scores.values)
    
    def test_weighted_edges(self):
        """Test PageRank with weighted edges."""
        graph = {
            "A": {"B": 2, "C": 1},
            "B": {"C": 3},
            "C": {"A": 1},
        }
        
        scores = power_iteration(graph)
        
        assert len(scores) == 3
        assert abs(scores.sum() - 1.0) < 1e-10
        assert all(score > 0 for score in scores.values)
    
    def test_custom_rsp(self):
        """Test PageRank with custom random surfer probability."""
        graph = {
            "A": {"B": 1},
            "B": {"A": 1},
        }
        
        scores_low_rsp = power_iteration(graph, rsp=0.1)
        scores_high_rsp = power_iteration(graph, rsp=0.9)
        
        assert len(scores_low_rsp) == 2
        assert len(scores_high_rsp) == 2
        assert abs(scores_low_rsp.sum() - 1.0) < 1e-10
        assert abs(scores_high_rsp.sum() - 1.0) < 1e-10
    
    def test_convergence_parameters(self):
        """Test PageRank with different convergence parameters."""
        graph = {
            "A": {"B": 1, "C": 1},
            "B": {"C": 1},
            "C": {"A": 1},
        }
        
        scores_strict = power_iteration(graph, epsilon=1e-8, max_iterations=2000)
        scores_loose = power_iteration(graph, epsilon=1e-3, max_iterations=100)
        
        assert len(scores_strict) == 3
        assert len(scores_loose) == 3
        assert abs(scores_strict.sum() - 1.0) < 1e-10
        assert abs(scores_loose.sum() - 1.0) < 1e-10
    
    def test_empty_graph_raises_error(self):
        """Test that empty graph raises ValueError."""
        with pytest.raises(ValueError, match="There must be at least one node"):
            power_iteration({})
    
    def test_list_input(self):
        """Test PageRank with list input format."""
        graph = [
            [0, 1, 1],
            [0, 0, 1],
            [1, 0, 0]
        ]
        
        scores = power_iteration(graph)
        
        assert len(scores) == 3
        assert abs(scores.sum() - 1.0) < 1e-10
        assert all(score > 0 for score in scores.values)
    
    def test_self_loops(self):
        """Test PageRank with self-loops."""
        graph = {
            "A": {"A": 1, "B": 1},
            "B": {"B": 1, "A": 1},
        }
        
        scores = power_iteration(graph)
        
        assert len(scores) == 2
        assert abs(scores.sum() - 1.0) < 1e-10
        assert all(score > 0 for score in scores.values)
    
    def test_deterministic_results(self):
        """Test that PageRank produces deterministic results."""
        graph = {
            "A": {"B": 1, "C": 1},
            "B": {"C": 1},
            "C": {"A": 1},
        }
        
        scores1 = power_iteration(graph)
        scores2 = power_iteration(graph)
        
        pd.testing.assert_series_equal(scores1, scores2)
    
    def test_large_graph_performance(self):
        """Test PageRank on a larger graph for performance."""
        nodes = [f"node_{i}" for i in range(50)]
        graph = {}
        
        for i, node in enumerate(nodes):
            graph[node] = {}
            for j in range(min(5, len(nodes))):
                target = nodes[(i + j + 1) % len(nodes)]
                graph[node][target] = 1
        
        scores = power_iteration(graph)
        
        assert len(scores) == 50
        assert abs(scores.sum() - 1.0) < 1e-10
        assert all(score > 0 for score in scores.values)
