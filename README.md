# pagerank

<div align="center">
  <a href="https://github.com/ashkonf/pagerank">
    <img src="https://via.placeholder.com/150/000000/FFFFFF/?text=PageRank" alt="PageRank Logo">
  </a>
  <h1>PageRank and TextRank</h1>
  <p>
    A lightweight, well-tested Python implementation of Google's PageRank algorithm and TextRank for keyword extraction.
  </p>
  <p>
    <a href="https://pypi.org/project/pagerank/">
      <img alt="PyPI" src="https://img.shields.io/pypi/v/pagerank.svg">
    </a>
    <a href="https://github.com/ashkonf/pagerank/actions/workflows/ci.yml">
      <img alt="CI" src="https://github.com/ashkonf/pagerank/actions/workflows/ci.yml/badge.svg">
    </a>
    <a href="https://codecov.io/gh/ashkonf/pagerank">
      <img alt="Codecov" src="https://codecov.io/gh/ashkonf/pagerank/graph/badge.svg?token=YOUR_TOKEN_HERE"/>
    </a>
    <a href="https://github.com/ashkonf/pagerank/blob/main/LICENSE">
      <img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-blue.svg"/>
    </a>
    <a href="https://github.com/ashkonf/pagerank/actions/workflows/ci.yml">
      <img alt="Pytest" src="https://img.shields.io/github/actions/workflow/status/ashkonf/pagerank/ci.yml?label=pytest&logo=pytest"/>
    </a>
    <a href="https://github.com/ashkonf/pagerank/actions/workflows/ci.yml">
      <img alt="Pyright" src="https://img.shields.io/github/actions/workflow/status/ashkonf/pagerank/ci.yml?label=pyright&logo=python"/>
    </a>
    <a href="https://results.pre-commit.ci/latest/github/ashkonf/pagerank/main">
      <img alt="pre-commit.ci status" src="https://results.pre-commit.ci/badge/github/ashkonf/pagerank/main.svg"/>
    </a>
  </p>
</div>

## Features

- **Lightweight PageRank**: A clean implementation of the PageRank algorithm using power iteration.
- **TextRank for SEO**: Extract meaningful keywords from text documents to understand topics and improve SEO.
- **Graph Flexibility**: Works with graphs represented as dictionaries or lists of lists.
- **Customizable**: Tweak parameters like damping factor, convergence tolerance, and max iterations.
- **Well-Tested**: High test coverage to ensure reliability.
- **Typed**: Fully type-hinted for better code quality and editor support.

## Table of Contents
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [PageRank Example](#pagerank-example)
  - [TextRank Example](#textrank-example)
- [Interactive Demo](#interactive-demo)
- [API Reference](#api-reference)
- [Development](#development)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

## Installation

This project uses `uv` for dependency management.

```bash
# Install only the required dependencies for production use
uv sync

# Install all development dependencies for contributing
uv sync --all-extras
```

You will also need to download NLTK data for TextRank. A helper script is provided:
```bash
uv run python download_nltk_data.py
```

## Quick Start

### PageRank Example

Calculate PageRank scores for a simple graph. The scores represent the "importance" of each node.

```python
from pagerank import power_iteration

# Define a graph where keys are nodes and values are outgoing links
graph = {
    "A": {"B": 1, "C": 1},
    "B": {"C": 1},
    "C": {"A": 1},
}

# Calculate PageRank scores
scores = power_iteration(graph)
print(scores)
# A    0.443029
# C    0.354423
# B    0.202548
# dtype: float64
```

### TextRank Example

Extract the most relevant keywords from a piece of text.

```python
from textrank import textrank

document = """
Natural language processing is a subfield of linguistics, computer science,
and artificial intelligence concerned with the interactions between computers
and human language.
"""

# Extract top keywords
keyword_scores = textrank(document)
print(keyword_scores.head())
# language       0.239396
# computer       0.177059
# intelligence   0.155705
# subfield       0.134703
# linguistics    0.134703
# dtype: float64
```

## Interactive Demo

For a more detailed walkthrough with visualizations, check out the Jupyter Notebook demo.

```bash
uv run jupyter notebook demo.ipynb
```

This demo covers:
- Basic and advanced PageRank examples.
- Keyword extraction with TextRank.
- Visualizing results with `matplotlib` and `seaborn`.

## API Reference

### `pagerank.power_iteration`

`power_iteration(transition_weights, rsp=0.15, epsilon=0.00001, max_iterations=1000)`

| Parameter | Type | Description | Default |
|---|---|---|---|
| `transition_weights` | `dict` or `list` | Graph representation. | Required |
| `rsp` | `float` | Random surfer probability (1 - damping). | `0.15` |
| `epsilon` | `float` | Convergence threshold. | `0.00001` |
| `max_iterations` | `int` | Max iterations to run. | `1000` |

**Returns:** `pandas.Series` with nodes as keys and PageRank scores as values.

### `textrank.textrank`

`textrank(document, window_size=2, rsp=0.15, relevant_pos_tags=["NN", "NNP", "ADJ"])`

| Parameter | Type | Description | Default |
|---|---|---|---|
| `document` | `str` | Text to analyze. | Required |
| `window_size` | `int` | Co-occurrence window size. | `2` |
| `rsp` | `float` | Random surfer probability. | `0.15` |
| `relevant_pos_tags`| `list[str]` | POS tags to consider as keywords. | `["NN", "NNP", "ADJ"]` |

**Returns:** `pandas.Series` with words as keys and TextRank scores as values, sorted descending.


## Development

To set up a development environment, clone the repo and install the dependencies.

```bash
git clone https://github.com/ashkonf/PageRank.git
cd PageRank
uv sync --all-extras
```

This project uses pre-commit hooks for quality checks. Install them with:
```bash
uv run pre-commit install
```

### Key Development Commands

- **Run tests**: `uv run pytest`
- **Check formatting**: `uv run ruff format --check .`
- **Check for linting errors**: `uv run ruff check .`
- **Run type checks**: `uv run pyright .`

## Roadmap

- [ ] **Summarization with TextRank**: Extend the library to support text summarization.
- [ ] **Performance Optimizations**: Investigate `scipy` sparse matrices for larger graphs.
- [ ] **More Graph Input Formats**: Support for `networkx` and other graph library objects.
- [ ] **Wheels for PyPI**: Publish pre-compiled wheels for faster installation.

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request.

1.  Fork the repository.
2.  Create your feature branch (`git checkout -b feature/my-new-feature`).
3.  Commit your changes (`git commit -am 'Add some feature'`).
4.  Push to the branch (`git push origin feature/my-new-feature`).
5.  Open a new Pull Request.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

---
_For more on the TextRank algorithm, see the [original paper](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf) by Mihalcea and Tarau._
