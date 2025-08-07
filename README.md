# PageRank

A Python 3 implementation of PageRank and TextRank.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
  - [PageRank](#pagerank)
  - [TextRank](#textrank)
- [Development](#development)
  - [Formatting and Linting](#formatting-and-linting)
  - [Static Type Checking](#static-type-checking)
  - [Running Tests](#running-tests)

## Installation
This project uses [uv](https://github.com/astral-sh/uv) for dependency management. After installing uv, dependencies will be installed automatically when running commands with `uv run`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Usage

### PageRank
```python
from pagerank import power_iteration

weights = {"A": {"B": 1.0}, "B": {"A": 1.0}}
print(power_iteration(weights))
```

### TextRank
Run the sample TextRank script on the provided stories:

```bash
uv run TextRank/textrank.py
```

## Development

### Formatting and Linting
Use [ruff](https://docs.astral.sh/ruff/) for formatting and linting:

```bash
uv run ruff format
uv run ruff check
```

### Static Type Checking
Run [pyright](https://github.com/microsoft/pyright) for type checking:

```bash
uv run pyright
```

### Running Tests
Tests are run with [pytest](https://pytest.org/) and require full coverage:

```bash
uv run pytest
```

The pre-commit configuration runs ruff, pyright and pytest together:

```bash
uv run pre-commit run --all-files
```
