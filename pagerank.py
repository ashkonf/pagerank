import math

import numpy
import pandas

# Generalized matrix operations:

def __extract_nodes(matrix):
    nodes = set()
    for col_key in matrix:
        nodes.add(col_key)
    for row_key in matrix.T:
        nodes.add(row_key)
    return nodes

def __make_square(matrix, keys, default=0.0):
    matrix = matrix.copy()
    
    def insert_missing_columns(matrix):
        for key in keys:
            if not key in matrix:
                matrix[key] = pandas.Series(default, index=matrix.index)
        return matrix

    matrix = insert_missing_columns(matrix) # insert missing columns
    matrix = insert_missing_columns(matrix.T).T # insert missing rows

    return matrix.fillna(default)

def __ensure_rows_positive(matrix):
    matrix = matrix.T
    for col_key in matrix:
        if matrix[col_key].sum() == 0.0:
            matrix[col_key] = pandas.Series(numpy.ones(len(matrix[col_key])), index=matrix.index)
    return matrix.T

def __normalize_rows(matrix):
    return matrix.div(matrix.sum(axis=1), axis=0)

def __euclidean_norm(series):
    return math.sqrt(series.dot(series))

# PageRank specific functionality:

def __start_state(nodes):
    if len(nodes) == 0: raise ValueError("There must be at least one node.")
    start_prob = 1.0 / float(len(nodes))
    return pandas.Series({node : start_prob for node in nodes})

def __integrate_random_surfer(nodes, transition_probabilities, rsp):
    alpha = 1.0 / float(len(nodes)) * rsp
    return transition_probabilities.copy().multiply(1.0 - rsp) + alpha

def power_iteration(transition_weights, rsp=0.15, epsilon=0.00001, max_iterations=1000):
    # Clerical work:
    transition_weights = pandas.DataFrame(transition_weights)
    nodes = __extract_nodes(transition_weights)
    transition_weights = __make_square(transition_weights, nodes, default=0.0)
    transition_weights = __ensure_rows_positive(transition_weights)

    # Setup:
    state = __start_state(nodes)
    transition_probabilities = __normalize_rows(transition_weights)
    transition_probabilities = __integrate_random_surfer(nodes, transition_probabilities, rsp)
    
    # Power iteration:
    for iteration in range(max_iterations):
        old_state = state.copy()
        state = state.dot(transition_probabilities)
        delta = state - old_state
        if __euclidean_norm(delta) < epsilon: 
            break

    return state
