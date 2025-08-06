# PageRank

A lightweight Python implementation of Google's PageRank algorithm with an example TextRank application for keyword extraction.

## Requirements

- Python 3.8+
- [NumPy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)

Install dependencies with:

```bash
pip install -r requirements.txt
```

## PageRank usage

```python
from pagerank import power_iteration

# edges represented as adjacency weights
graph = {
    "A": {"B": 1, "C": 1},
    "B": {"C": 1},
    "C": {"A": 1},
}

scores = power_iteration(graph)
print(scores)
```

## TextRank demo

Run the bundled TextRank example to extract keywords from sample stories:

```bash
python TextRank/textrank.py
```

## Development

Run lint checks and tests before committing:

```bash
python -m pyflakes pagerank.py TextRank/textrank.py
python -m pytest
```

## License

Apache-2.0

