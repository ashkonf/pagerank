from __future__ import annotations

import collections
import os
from typing import DefaultDict, Iterable, List

import nltk
import pandas as pd

import pagerank


# TextRank #####################################################################################


def __preprocess_document(document: str, relevant_pos_tags: Iterable[str]) -> List[str]:
    """Tokenize a document and filter words by part of speech."""

    words = __tokenize_words(document)
    pos_tags = __tag_parts_of_speech(words)

    filtered_words: List[str] = []
    for index, word in enumerate(words):
        word_lower = word.lower()
        tag = pos_tags[index]
        if not __is_punctuation(word_lower) and tag in relevant_pos_tags:
            filtered_words.append(word_lower)

    return filtered_words


def textrank(
    document: str,
    window_size: int = 2,
    rsp: float = 0.15,
    relevant_pos_tags: Iterable[str] | None = None,
) -> pd.Series:
    """Apply the TextRank algorithm to a document."""

    if relevant_pos_tags is None:
        relevant_pos_tags = ["NN", "ADJ"]

    words = __preprocess_document(document, relevant_pos_tags)

    edge_weights: DefaultDict[str, DefaultDict[str, float]] = collections.defaultdict(
        lambda: collections.defaultdict(float)
    )
    for index, word in enumerate(words):
        for other_index in range(index - window_size, index + window_size + 1):
            if 0 <= other_index < len(words) and other_index != index:
                other_word = words[other_index]
                edge_weights[word][other_word] += 1.0

    word_probabilities = pagerank.power_iteration(dict(edge_weights), rsp=rsp)
    word_probabilities.sort_values(ascending=False)

    return word_probabilities


# NLP utilities ################################################################################


def __ascii_only(string: str) -> str:
    return "".join([char if ord(char) < 128 else "" for char in string])


def __is_punctuation(word: str) -> bool:
    return word in [".", "?", "!", ",", '"', ":", ";", "'", "-"]


def __tag_parts_of_speech(words: Iterable[str]) -> List[str]:
    return [pair[1] for pair in nltk.pos_tag(list(words))]


def __tokenize_words(sentence: str) -> List[str]:
    return nltk.tokenize.word_tokenize(sentence)


# tests ########################################################################################


def apply_text_tank(file_name: str, title: str = "a document") -> None:
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
    apply_text_tank("Cinderalla.txt", "Cinderalla")
    apply_text_tank("Beauty_and_the_Beast.txt", "Beauty and the Beast")
    apply_text_tank("Rapunzel.txt", "Rapunzel")


if __name__ == "__main__":
    main()
