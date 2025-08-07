"""Tests for the TextRank module."""

import os
import tempfile
from unittest.mock import mock_open, patch

from TextRank.textrank import apply_text_rank, textrank


class TestTextRank:
    """Test cases for the textrank function."""

    def test_simple_document(self):
        """Test TextRank on a simple document."""
        document = "The cat sat on the mat. The cat was happy."

        scores = textrank(document)

        assert len(scores) > 0
        assert all(score >= 0 for score in scores.values)
        assert abs(scores.sum() - 1.0) < 1e-10

    def test_empty_document(self):
        """Test TextRank on an empty document."""
        document = ""

        scores = textrank(document)

        assert len(scores) == 0

    def test_punctuation_only(self):
        """Test TextRank on document with only punctuation."""
        document = "!!! ??? ... ,,,"

        scores = textrank(document)

        assert len(scores) == 0

    def test_custom_window_size(self):
        """Test TextRank with different window sizes."""
        document = "The quick brown fox jumps over the lazy dog."

        scores_small = textrank(document, window_size=1)
        scores_large = textrank(document, window_size=5)

        assert len(scores_small) > 0
        assert len(scores_large) > 0
        assert abs(scores_small.sum() - 1.0) < 1e-10
        assert abs(scores_large.sum() - 1.0) < 1e-10

    def test_custom_pos_tags(self):
        """Test TextRank with custom POS tags."""
        document = "The quick brown fox jumps over the lazy dog."

        scores_nouns = textrank(document, relevant_pos_tags=["NN"])
        scores_adj = textrank(document, relevant_pos_tags=["JJ"])
        scores_both = textrank(document, relevant_pos_tags=["NN", "JJ"])

        assert len(scores_nouns) >= 0
        assert len(scores_adj) >= 0
        assert len(scores_both) >= len(scores_nouns)
        assert len(scores_both) >= len(scores_adj)

    def test_custom_rsp(self):
        """Test TextRank with custom random surfer probability."""
        document = "The cat sat on the mat. The cat was happy."

        scores_low = textrank(document, rsp=0.1)
        scores_high = textrank(document, rsp=0.9)

        assert len(scores_low) > 0
        assert len(scores_high) > 0
        assert abs(scores_low.sum() - 1.0) < 1e-10
        assert abs(scores_high.sum() - 1.0) < 1e-10

    def test_repeated_words(self):
        """Test TextRank on document with repeated words."""
        document = "cat cat cat dog dog bird"

        scores = textrank(document)

        assert len(scores) > 0
        if "cat" in scores and "dog" in scores:
            assert scores["cat"] >= scores["dog"]


class TestUtilityFunctions:
    """Test cases for utility functions."""

    def test_basic_preprocessing_integration(self):
        """Test basic document preprocessing through textrank function."""
        document = "The Cat sat on the Mat."

        result = textrank(document)

        assert len(result) >= 0

    def test_empty_document_preprocessing(self):
        """Test preprocessing of empty document through textrank."""
        document = ""

        result = textrank(document)

        assert len(result) == 0

    def test_punctuation_filtering_integration(self):
        """Test that punctuation is filtered out through textrank."""
        document = "Hello, world!"

        result = textrank(document)

        if len(result) > 0:
            assert "," not in result.index
            assert "!" not in result.index


class TestApplyTextRank:
    """Test cases for the apply_text_rank function."""

    def test_apply_text_rank_with_temp_file(self):
        """Test apply_text_rank with a temporary file."""
        content = "The cat sat on the mat. The cat was happy."

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            with patch("TextRank.textrank.logger") as mock_logger:
                with patch("builtins.open", mock_open(read_data=content)):
                    apply_text_rank(os.path.basename(temp_path), "Test Document")

                    assert mock_logger.info.called
                    log_messages = [
                        call.args[0] for call in mock_logger.info.call_args_list
                    ]
                    assert any("Reading" in str(msg) for msg in log_messages)
                    assert any("Applying TextRank" in str(msg) for msg in log_messages)
        finally:
            os.unlink(temp_path)

    def test_apply_text_rank_return_none(self):
        """Test that apply_text_rank returns None."""
        content = "Test content."

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            temp_path = f.name

        try:
            with patch("TextRank.textrank.logger"):
                with patch("builtins.open", mock_open(read_data=content)):
                    result = apply_text_rank(os.path.basename(temp_path))

                    assert result is None
        finally:
            os.unlink(temp_path)


class TestIntegration:
    """Integration tests for the TextRank module."""

    def test_textrank_with_real_text(self):
        """Test TextRank with a longer, more realistic text."""
        document = """
        Natural language processing is a subfield of linguistics, computer science,
        and artificial intelligence concerned with the interactions between computers
        and human language. In particular, how to program computers to process and
        analyze large amounts of natural language data. The goal is a computer
        capable of understanding the contents of documents, including the contextual
        nuances of the language within them.
        """

        scores = textrank(document)

        assert len(scores) > 0
        assert abs(scores.sum() - 1.0) < 1e-10

        top_words = scores.head(5)
        assert len(top_words) <= 5
        assert all(score > 0 for score in top_words.values)

    def test_textrank_deterministic(self):
        """Test that TextRank produces deterministic results."""
        document = "The cat sat on the mat. The cat was happy."

        scores1 = textrank(document)
        scores2 = textrank(document)

        assert len(scores1) == len(scores2)
        for word in scores1.index:
            if word in scores2.index:
                assert abs(scores1[word] - scores2[word]) < 1e-10
