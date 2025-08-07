import collections
import os
import sys
from typing import Any, List

import nltk

sys.path.append(".")

import pagerank

"""
    textrank.py
    -----------
    This module implements TextRank, an unsupervised keyword
    significance scoring algorithm. TextRank builds a weighted
    graph representation of a document using words as nodes
    and co-occurrence frequencies between pairs of words as edge
    weights. It then applies PageRank to this graph, and
    treats the PageRank score of each word as its significance.
    The original research paper proposing this algorithm is
    available here:

        https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf
"""

## TextRank #####################################################################################


def __preprocess_document(document: str, relevant_pos_tags: List[str]) -> List[str]:
    """Preprocess a document by tokenizing and filtering words by part-of-speech tags.

    This function accepts a string representation of a document as input,
    and returns a tokenized list of words corresponding to that document,
    filtered by relevant parts of speech.

    Args:
        document: The input document as a string.
        relevant_pos_tags: List of POS tags to keep (e.g., ["NN", "ADJ"]).

    Returns:
        A list of lowercase words filtered by relevant POS tags.
    """
    words = __tokenize_words(document)
    pos_tags = __tag_parts_of_speech(words)

    filtered_words = []
    for index, word in enumerate(words):
        word = word.lower()
        tag = pos_tags[index]
        if not __is_punctuation(word) and tag in relevant_pos_tags:
            filtered_words.append(word)

    return filtered_words


def textrank(
    document: str,
    window_size: int = 2,
    rsp: float = 0.15,
    relevant_pos_tags: List[str] | None = None,
) -> Any:
    """Apply TextRank algorithm to extract keyword significance scores from a document.

    This function implements the TextRank algorithm by creating a graph representation
    of the document using words as nodes and co-occurrence frequencies between pairs
    of words as edge weights. It then applies PageRank to this graph to determine
    the significance of each word.

    Args:
        document: A string representing the input document. All characters should be
            standard ASCII to avoid exceptions.
        window_size: Width of the window in which two words must fall to be considered
            co-occurring.
        rsp: Random surfer probability that controls the chance of jumping to any node.
        relevant_pos_tags: Parts of speech to consider; by default nouns and adjectives.

    Returns:
        A pandas Series (that can be treated as a dictionary) that maps words in the
        document to their associated TextRank significance scores. Note that only words
        that are classified as having relevant POS tags are present in the result.

    Example:
        >>> document = "The cat sat on the mat. The cat was happy."
        >>> scores = textrank(document)
        >>> print(scores.head())
    """
    if relevant_pos_tags is None:
        relevant_pos_tags = ["NN", "ADJ"]

    # Tokenize document:
    words = __preprocess_document(document, relevant_pos_tags)

    if not words:
        import pandas

        return pandas.Series(dtype=float)

    # Build a weighted graph where nodes are words and
    # edge weights are the number of times words cooccur
    # within a window of predetermined size. In doing so
    # we double count each co-occurrence, but that will not
    # alter relative weights which ultimately determine
    # TextRank scores.
    edge_weights = collections.defaultdict(lambda: collections.Counter())
    for index, word in enumerate(words):
        for other_index in range(index - window_size, index + window_size + 1):
            if other_index >= 0 and other_index < len(words) and other_index != index:
                other_word = words[other_index]
                edge_weights[word][other_word] += 1

    # Apply PageRank to the weighted graph:
    edge_weights_dict = {
        word: {other_word: float(count) for other_word, count in counter.items()}
        for word, counter in edge_weights.items()
    }

    if not edge_weights_dict:
        import pandas

        return pandas.Series(dtype=float)

    word_probabilities = pagerank.power_iteration(edge_weights_dict, rsp=rsp)
    word_probabilities.sort_values(ascending=False)

    return word_probabilities


## NLP utilities ################################################################################


def __ascii_only(string: str) -> str:
    """Filter string to contain only ASCII characters.

    Args:
        string: The input string to filter.

    Returns:
        A string containing only ASCII characters.
    """
    return "".join([char if ord(char) < 128 else "" for char in string])


def __is_punctuation(word: str) -> bool:
    """Check if a word is punctuation.

    Args:
        word: The word to check.

    Returns:
        True if the word is punctuation, False otherwise.
    """
    return word in [".", "?", "!", ",", '"', ":", ";", "'", "-"]


def __tag_parts_of_speech(words: List[str]) -> List[str]:
    """Tag parts of speech for a list of words.

    Args:
        words: List of words to tag.

    Returns:
        List of POS tags corresponding to the input words.
    """
    return [pair[1] for pair in nltk.pos_tag(words)]


def __tokenize_words(sentence: str) -> List[str]:
    """Tokenize a sentence into words.

    Args:
        sentence: The input sentence to tokenize.

    Returns:
        List of tokenized words.
    """
    return nltk.tokenize.word_tokenize(sentence)


## tests ########################################################################################


def apply_text_rank(file_name: str, title: str = "a document") -> None:
    """Apply TextRank algorithm to a text file and print results.

    This function is a wrapper around the textrank function. It accepts a plain text
    document as its input, transforms that document into the data format expected by
    the textrank function, calls textrank to perform the algorithm, and prints out
    the results along with progress indicators.

    Args:
        file_name: Name or full path of the file that contains the document the
            TextRank algorithm will be applied to.
        title: The document's title, used only in printed progress indicators.

    Returns:
        None. This function prints its results rather than returning them.
    """
    print()
    print(f'Reading "{title}" ...')
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    document = open(file_path).read()
    document = __ascii_only(document)

    print(f'Applying TextRank to "{title}" ...')
    keyword_scores = textrank(document)

    print()
    header = f'Keyword Significance Scores for "{title}":'
    print(header)
    print("-" * len(header))
    print(keyword_scores)
    print()


def main() -> None:
    """Run TextRank on sample fairy tale documents."""
    apply_text_rank("Cinderella.txt", "Cinderella")
    apply_text_rank("Beauty_and_the_Beast.txt", "Beauty and the Beast")
    apply_text_rank("Rapunzel.txt", "Rapunzel")


if __name__ == "__main__":
    main()
