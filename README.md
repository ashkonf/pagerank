# PageRank
A Python implementation of Google's famous PageRank algorithm.

## Setup

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

The pagerank module exports one public function:

     power_iteration(transition_weights, rsp=0.15, epsilon=0.00001, max_iterations=1000)
     
This function applies the PageRank algorithm to a provided graph to determine the steady probabilities with which a random walk through the graph will end up at each node. It does so using power iteration, an algorithm approximating steady state probabilities by iteratively improving them until convergence. This algorithm is guaranteed to converege to the correct steady state probabilities for ergodic Markov chains, which PageRank graphs are.
  
Arguments:

| Name              | Type                                        | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Optional? | Default Value |
|-------------------|---------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------|---------------|
| `transition_weights` | Once nested list or dictionary, or similar. | The graph to which the PageRank algorithm is applied represented as the weights between its nodes. It should be encoded sparsely as a once nested dictionary or a once nested list. If encoded sparsely as a once nested dictionary, keys and nested keys should correspond to node names and values to weights. Other datastructures with the same nested key-value interface, such as certain Pandas matrices, are also acceptable. If encoded as a once nested list, node names are assigned as the indices corresponding to locations in the nested list. | False    |            |
| `rsp`               | `float`                                       | The graph to which the PageRank algorithm is applied represented as the weights between its nodes. It should be encoded sparsely as a once nested dictionary or a once nested list. If encoded sparsely as a once nested dictionary, keys and nested keys should correspond to node names and values to weights. Other datastructures with the same nested key-value interface, such as certain Pandas matrices, are also acceptable. If encoded as a once nested list, node names are assigned as the indices corresponding to locations in the nested list. | True     | `0.15`          |
| `epsilon`           | `float`                                       | The threshold of convergence. If the Euclidean norm of the difference between the approximations of the steady state vector before and after an iteration of power iteration is smaller than epsilon, the algorithm will consider itself to have converged and will terminate.                                                                                                                                                                                                                                                                                | True     | `0.00001`       |
| `max_iterations`     | `int`                                         | The number of iterations after which power iteration will be terminated even if it has not yet converged.                                                                                                                                                                                                                                                                                                                                                                                                                                                     | True     | `1000`          |

Note that elements of `transition_weights` need not be probabilities (meaning its rows need not be normalized), and the random surfer probabilities should not be incorporated into it. The `power_iteration` function will perform normalization and integration of the random surfer probabilities.

Return value: This function returns a Pandas series whose keys are node names and whose values are the corresponding steady state probabilities. This Pandas series can be treated as a dict.

## Example Usage: TextRank

An implementation of TextRank and three stories one can apply it to are included as a sample usage of the PageRank module. TextRank is an unsupervised keyword significance scoring algorithm that applies PageRank to a graph built from words found in a document to determine the significance of each word. The textrank module, located in the TextRank directory, implements the TextRank algorithm.

The textrank module's main method applies TextRank to three fairy tales, Rapunzel, Cinderalla and Beauty and the Beast. It then prints out the results, an ordered list of keywords and their associated significance scores. To run this example, simply navigate to the TextRank directory and run textrank.py:

     uv run python TextRank/textrank.py
     
For more information about TextRank, see the [original paper](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf) that proposed it.

### TextRank Implementation

The textrank module also exports two public functions:

     textrank(document, window_size=2, rsp=0.15, relevant_pos_tags=["NN", "ADJ"])
     apply_text_rank(file_name, title="a document")

#### Function: textrank

     textrank(document, window_size=2, rsp=0.15, relevant_pos_tags=["NN", "ADJ"])

The textrank function implements the TextRank algorithm. It creates a graph representing the document provided to it as an argument, applies the PageRank algorithm to that graph, and returns a list of words in the document sorted in descending order of resulting node weights. The graph representing the document is created using the words found in the document as nodes, and the frequency with which words co-occur in close proximity as weights. The node weights provided by the PageRank algorithm are considered to be the words' significance in the document.

Arguments:

| Name            | Type      | Description                                                                                                                                                                                                                                                                    | Optional? | Default Value |
|-----------------|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|---------------|
| `document`        | `str`       | A string representing a document. Note that all characters in the string must be standard ASCII characters to avoid exceptions.                                                                                                                                                | False     |            |
| `windowSize`      | `int`       | The width of the window in which two words must fall to be considered to have co-occured. For a window size of 2, the default value of this argument, a word will be considered to have co-occurred with any word one or two words away from it in the document.               | True      | `2`             |
| `rsp`             | `float`     | Again the random surfer probability that represents the probability with which the random walk through the graph will deviate from its edges and instead jump randomly to any node in the graph.                                                                               | True      | `0.15`          |
| `relevant_pos_tags` | `[str]` | The TextRank algorithm will filter the words in a document down to only those of certain parts of speech. The default implementation of the algorithm only considers nouns and adjectives. See the original paper that proposed TextRank for a justification of this decision. | True      | `["NN", "ADJ"]` |


Return Value: This function returns a list of words found in the document (filtered by parts of speech) in descending order of node weights.

#### Function: apply_text_rank

     apply_text_rank(file_name, title="a document")

The applyTextRank function is a wrapper around the textrank function. It accepts a plain text document as its input, transfors that document into the data format expected by the textrank function, calls the textrank function to perform the textrank algorithm, and prints out the result cleanly along with a few helpful progress indicators.

Arguments:

| Name     | Type                                        | Description                                                                                             | Optional? | Default Value |
|----------|---------------------------------------------|---------------------------------------------------------------------------------------------------------|----------|---------------|
| `file_name` | Once nested list or dictionary, or similar. | The name or full path of the file that contains the document the TextRank algorithm will be applied to. | False    |            |
| `title`    | `str`                                         | The document's title, an optional argument used only in printed progress indicators.                    | True     | `"a document"`  |

Return value: This function has no return value, and instead prints out its results.

If you would like to apply TextRank to a story or document of your choosing, add a plain text file containing the story to the TextRank directory and call the applyTextRank function, passing in the name of the file and optionally the document's title.
