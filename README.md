# PageRank
A Python implementation of Google's famous PageRank algorithm.

## Table of Contents
- [Setup](#setup)
  - [Dependencies](#dependencies)
- [Usage](#usage)
- [Example Usage: TextRank](#example-usage-textrank)
  - [TextRank Implementation](#textrank-implementation)
    - [Function: textrank](#function-textrank)
    - [Function: apply_text_rank](#function-apply-text-rank)

This project targets Python 3 and uses [uv](https://github.com/astral-sh/uv) for
dependency management. To create a virtual environment and install
dependencies, run:

```
uv sync
```

You can then execute modules with `uv run`, for example:

```
uv run python TextRank/textrank.py
```

### Dependencies

This module relies on a few commonly used Python libraries:

1.  [Numpy](http://www.numpy.org/)
2.  [Pandas](http://pandas.pydata.org/)
3.  [NLTK](https://www.nltk.org/)

## Usage
The `pagerank` module exports one public function:

```python
power_iteration(transition_weights, rsp=0.15, epsilon=0.00001, max_iterations=1000)
```

This function applies the PageRank algorithm to a provided graph to determine the steady probabilities with which a random walk through the graph will end up at each node. It uses power iteration, an algorithm that iteratively refines the steady state probabilities until convergence. This algorithm is guaranteed to converge to the correct probabilities for ergodic Markov chains, which PageRank graphs are.

### Arguments

| Name | Type | Description | Optional? | Default |
|------|------|-------------|-----------|---------|
| `transition_weights` | `dict` or `list` | Sparse representation of the graph as nested dicts or lists. Keys correspond to node names and values to weights. | No | — |
| `rsp` | `float` | Random surfer probability controlling the chance of jumping to any node. | Yes | `0.15` |
| `epsilon` | `float` | Threshold of convergence; iteration stops when successive approximations are closer than this value. | Yes | `0.00001` |
| `max_iterations` | `int` | Maximum number of iterations before termination even without convergence. | Yes | `1000` |

Note that elements of `transition_weights` need not be probabilities (rows need not be normalized), and the random surfer probabilities should not be incorporated into it. The `power_iteration` function will perform normalization and integrate the random surfer probabilities.

Return value: This function returns a Pandas series whose keys are node names and whose values are the corresponding steady state probabilities. This series can be treated as a dict.

## Example Usage: TextRank
An implementation of TextRank and three sample stories are included as a demonstration of the PageRank module. TextRank is an unsupervised keyword significance scoring algorithm that applies PageRank to a graph built from words found in a document to determine the significance of each word. The `textrank` module, located in the `TextRank` directory, implements this algorithm.

The module's main method applies TextRank to three fairy tales—Rapunzel, Cinderella and Beauty and the Beast—and prints out the results. To run this example, simply navigate to the `TextRank` directory and run:

```bash
     uv run python TextRank/textrank.py
```

For more information about TextRank, see the [original paper](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf) that proposed it.

### TextRank Implementation
The `textrank` module also exports two public functions:

```python
textrank(document, window_size=2, rsp=0.15, relevant_pos_tags=["NN", "ADJ"])
apply_text_rank(file_name, title="a document")
```

#### Function: textrank

```python
textrank(document, window_size=2, rsp=0.15, relevant_pos_tags=["NN", "ADJ"])
```

The `textrank` function implements the TextRank algorithm. It creates a graph representing the document provided to it as an argument, applies the PageRank algorithm to that graph, and returns a list of words in the document sorted in descending order of node weights. The graph representing the document is created using the words found in the document as nodes and the frequency with which words co-occur in close proximity as weights.

Arguments:

| Name | Type | Description | Optional? | Default |
|------|------|-------------|-----------|---------|
| `document` | `str` | A string representing a document. All characters must be standard ASCII to avoid exceptions. | No | — |
| `window_size` | `int` | Width of the window in which two words must fall to be considered co-occurring. | Yes | `2` |
| `rsp` | `float` | Random surfer probability that controls the chance of jumping to any node. | Yes | `0.15` |
| `relevant_pos_tags` | `[str]` | Parts of speech to consider; by default nouns and adjectives. | Yes | `["NN", "ADJ"]` |

Return Value: This function returns a list of words found in the document (filtered by parts of speech) in descending order of node weights.

#### Function: apply_text_rank

```python
apply_text_rank(file_name, title="a document")
```

The `apply_text_rank` function is a wrapper around the `textrank` function. It accepts a plain text document as its input, transforms that document into the data format expected by the `textrank` function, calls `textrank` to perform the algorithm, and prints out the results along with progress indicators.

Arguments:

| Name | Type | Description | Optional? | Default |
|------|------|-------------|-----------|---------|
| `file_name` | `str` | Name or full path of the file that contains the document the TextRank algorithm will be applied to. | No | — |
| `title` | `str` | The document's title, used only in printed progress indicators. | Yes | "a document" |

Return value: This function has no return value, and instead prints out its results.

If you would like to apply TextRank to a story or document of your choosing, add a plain text file containing the story to the `TextRank` directory and call the `apply_text_rank` function, passing in the name of the file and optionally the document's title.
