import sys
from pathlib import Path

import nltk
import pytest
import runpy

sys.path.append(str(Path(__file__).resolve().parents[1]))
import TextRank.textrank as textrank

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("averaged_perceptron_tagger_eng", quiet=True)


def test_ascii_only() -> None:
    assert textrank.__ascii_only("héllo!") == "hllo!"


def test_is_punctuation() -> None:
    assert textrank.__is_punctuation("!")
    assert not textrank.__is_punctuation("a")


def test_tokenize_and_tag() -> None:
    words = textrank.__tokenize_words("Hello world!")
    assert words == ["Hello", "world", "!"]
    tags = textrank.__tag_parts_of_speech(["Hello", "world"])
    assert tags == ["NNP", "NN"]


def test_preprocess_document() -> None:
    result = textrank.__preprocess_document("Hello world!", ["NNP", "NN"])
    assert result == ["hello", "world"]


def test_textrank_small() -> None:
    document = "Hello world hello"
    scores = textrank.textrank(
        document, window_size=1, rsp=0.15, relevant_pos_tags=["NNP", "NN"]
    )
    assert set(scores.index) == {"hello", "world"}
    assert scores.sum() == pytest.approx(1.0)


def test_apply_text_tank_and_main() -> None:
    textrank.apply_text_tank("Cinderalla.txt", "Cinderalla")
    textrank.main()


def test_module_entry_point() -> None:
    runpy.run_module("TextRank.textrank", run_name="__main__")
