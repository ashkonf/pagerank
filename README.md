# PageRank

A lightweight Python implementation of Google's PageRank algorithm with an example TextRank application for keyword extraction.

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [PageRank Example](#pagerank-example)
  - [TextRank Example](#textrank-example)
- [PageRank Usage](#pagerank-usage)
- [TextRank Demo](#textrank-demo)
- [Development](#development)
  - [Setting Up Development Environment](#setting-up-development-environment)
  - [Running Tests](#running-tests)
  - [Code Quality Tools](#code-quality-tools)
  - [Pre-commit Hooks](#pre-commit-hooks)
- [API Reference](#api-reference)
  - [PageRank Module](#pagerank-module)
  - [TextRank Module](#textrank-module)
- [Examples](#examples)
  - [Basic PageRank](#basic-pagerank)
  - [Advanced PageRank with Custom Parameters](#advanced-pagerank-with-custom-parameters)
  - [TextRank for Keyword Extraction](#textrank-for-keyword-extraction)
  - [Custom TextRank Analysis](#custom-textrank-analysis)
- [Contributing](#contributing)
- [License](#license)

## Installation

This project requires Python 3.8+ and uses [uv](https://github.com/astral-sh/uv) for dependency management.

To install dependencies:

```bash
uv sync
```

For development with additional tools:

```bash
uv sync --all-extras
```

## Quick Start

### PageRank Example

```python
from pagerank import power_iteration

# Define a graph as adjacency weights
graph = {
    "A": {"B": 1, "C": 1},
    "B": {"C": 1},
    "C": {"A": 1},
}

scores = power_iteration(graph)
print(scores)
```

### TextRank Example

```python
from TextRank import textrank

document = "The cat sat on the mat. The cat was happy."
keyword_scores = textrank(document)
print(keyword_scores.head())
```

## PageRank Usage

The `pagerank` module provides a Python implementation of Google's PageRank algorithm using power iteration. It can handle both dictionary and list representations of graphs.

```python
from pagerank import power_iteration

# Using dictionary format (recommended)
graph = {
    "Page1": {"Page2": 1, "Page3": 2},
    "Page2": {"Page3": 1},
    "Page3": {"Page1": 1},
}

scores = power_iteration(graph, rsp=0.15, epsilon=0.00001, max_iterations=1000)
```

## TextRank Demo

Run the bundled TextRank example to extract keywords from sample fairy tales:

```bash
uv run python TextRank/textrank.py
```

This will analyze three classic fairy tales (Cinderella, Beauty and the Beast, and Rapunzel) and display keyword significance scores.

## Development

### Setting Up Development Environment

1. Clone the repository:
```bash
git clone https://github.com/ashkonf/PageRank.git
cd PageRank
```

2. Install dependencies with development tools:
```bash
uv sync --all-extras
```

3. Download required NLTK data:
```bash
uv run python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

### Running Tests

Run the full test suite with coverage:

```bash
uv run pytest
```

Run tests with detailed coverage report:

```bash
uv run pytest --cov=. --cov-report=term-missing --cov-report=html
```

Run specific test files:

```bash
uv run pytest tests/test_pagerank.py
uv run pytest tests/test_textrank.py
```

### Code Quality Tools

This project uses several code quality tools that can be run individually:

#### Ruff (Linting and Formatting)

Check for linting issues:
```bash
uv run ruff check .
```

Auto-fix linting issues:
```bash
uv run ruff check . --fix
```

Format code:
```bash
uv run ruff format .
```

Check formatting without making changes:
```bash
uv run ruff format . --check
```

#### Pyright (Type Checking)

Run type checking:
```bash
uv run pyright .
```

Run type checking on specific files:
```bash
uv run pyright pagerank.py
uv run pyright TextRank/textrank.py
```

### Pre-commit Hooks

Install pre-commit hooks to automatically run quality checks before commits:

```bash
uv run pre-commit install
```

Run pre-commit hooks manually:
```bash
uv run pre-commit run --all-files
```

The pre-commit configuration runs:
- `ruff check` with auto-fix
- `ruff format` for code formatting
- `pyright` for type checking
- `pytest` for running tests

## API Reference

### PageRank Module

#### `power_iteration(transition_weights, rsp=0.15, epsilon=0.00001, max_iterations=1000)`

Applies the PageRank algorithm to determine steady-state probabilities for a graph.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `transition_weights` | `dict` or `list` | Graph representation as nested dicts or lists. Keys are node names, values are edge weights. | Required |
| `rsp` | `float` | Random surfer probability (1 - damping factor). Controls probability of jumping to any node. | `0.15` |
| `epsilon` | `float` | Convergence threshold. Iteration stops when successive approximations differ by less than this value. | `0.00001` |
| `max_iterations` | `int` | Maximum iterations before termination, even without convergence. | `1000` |

**Returns:**
- `pandas.Series`: Node names as keys, steady-state probabilities as values. Can be treated as a dictionary.

**Example:**
```python
graph = {"A": {"B": 1}, "B": {"A": 1}}
scores = power_iteration(graph)
print(f"Node A score: {scores['A']}")
```

### TextRank Module

#### `textrank(document, window_size=2, rsp=0.15, relevant_pos_tags=["NN", "ADJ"])`

Implements TextRank algorithm for keyword extraction from documents.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `document` | `str` | Input document text. Should contain standard ASCII characters. | Required |
| `window_size` | `int` | Window size for word co-occurrence. Words within this distance are considered connected. | `2` |
| `rsp` | `float` | Random surfer probability for PageRank algorithm. | `0.15` |
| `relevant_pos_tags` | `list[str]` | Part-of-speech tags to include. Common values: "NN" (nouns), "JJ" (adjectives). | `["NN", "ADJ"]` |

**Returns:**
- `pandas.Series`: Words as keys, significance scores as values, sorted by score (descending).

#### `apply_text_rank(file_name, title="a document")`

Convenience function to apply TextRank to a text file and print results.

**Parameters:**

| Name | Type | Description | Default |
|------|------|-------------|---------|
| `file_name` | `str` | Path to text file (relative to TextRank directory). | Required |
| `title` | `str` | Document title for display purposes. | `"a document"` |

**Returns:**
- `None`: Prints results to console.

## Examples

### Basic PageRank

```python
from pagerank import power_iteration

# Simple three-node graph
graph = {
    "A": {"B": 1, "C": 1},
    "B": {"C": 1},
    "C": {"A": 1},
}

scores = power_iteration(graph)
print("PageRank scores:")
for node, score in scores.items():
    print(f"{node}: {score:.4f}")
```

### Advanced PageRank with Custom Parameters

```python
# Weighted graph with custom parameters
weighted_graph = {
    "Home": {"About": 2, "Products": 3, "Contact": 1},
    "About": {"Home": 1, "Products": 1},
    "Products": {"Home": 1, "About": 1, "Contact": 2},
    "Contact": {"Home": 1},
}

scores = power_iteration(
    weighted_graph,
    rsp=0.1,  # Lower random surfer probability
    epsilon=1e-8,  # Higher precision
    max_iterations=2000
)
```

### TextRank for Keyword Extraction

```python
from TextRank import textrank

# Analyze a document
document = """
Natural language processing is a subfield of computer science and artificial intelligence.
It focuses on the interaction between computers and human language. The goal is to enable
computers to understand, interpret, and generate human language in a valuable way.
"""

# Extract keywords (nouns and adjectives)
keywords = textrank(document, window_size=3, relevant_pos_tags=["NN", "JJ"])

print("Top 10 keywords:")
for word, score in keywords.head(10).items():
    print(f"{word}: {score:.4f}")
```

### Custom TextRank Analysis

```python
# Focus only on nouns with larger window
noun_keywords = textrank(
    document,
    window_size=5,
    relevant_pos_tags=["NN", "NNS", "NNP", "NNPS"],  # All noun types
    rsp=0.2
)

# Focus only on adjectives
adj_keywords = textrank(
    document,
    window_size=2,
    relevant_pos_tags=["JJ", "JJR", "JJS"],  # All adjective types
    rsp=0.15
)
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run the test suite: `uv run pytest`
5. Run code quality checks: `uv run pre-commit run --all-files`
6. Commit your changes: `git commit -am 'Add feature'`
7. Push to the branch: `git push origin feature-name`
8. Create a Pull Request

Please ensure all tests pass and code quality checks are satisfied before submitting a PR.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

---

**Dependencies:**
- [NumPy](https://numpy.org/) - Numerical computing
- [Pandas](https://pandas.pydata.org/) - Data manipulation and analysis
- [NLTK](https://www.nltk.org/) - Natural language processing

For more information about the TextRank algorithm, see the [original paper](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf) by Mihalcea and Tarau.

