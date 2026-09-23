"""Tests for data preprocessing utilities."""

import numpy as np
import pytest
import tensorflow as tf

from src.data.preprocessing import (
    unicode_to_ascii,
    preprocess_sentence,
    create_tokenizer,
    tokenize_and_pad,
    get_vocab_size,
)


class TestUnicodeToAscii:
    """Tests for unicode_to_ascii function."""

    def test_basic_ascii(self):
        """ASCII text should pass through unchanged."""
        assert unicode_to_ascii("hello") == "hello"

    def test_german_umlaut_a(self):
        """German ä with combining diacritic should convert to a."""
        # Note: This tests the NFD normalization path
        assert unicode_to_ascii("ä") == "a"

    def test_accent_removal(self):
        """Accented characters should have accents removed."""
        assert unicode_to_ascii("café") == "cafe"
        assert unicode_to_ascii("naïve") == "naive"

    def test_mixed_text(self):
        """Mixed ASCII and unicode should work correctly."""
        assert unicode_to_ascii("hello wörld") == "hello world"


class TestPreprocessSentence:
    """Tests for preprocess_sentence function."""

    def test_lowercase(self):
        """Text should be lowercased."""
        assert preprocess_sentence("HELLO") == "hello"

    def test_strip_whitespace(self):
        """Leading and trailing whitespace should be stripped."""
        assert preprocess_sentence("  hello  ") == "hello"

    def test_german_umlaut_replacement(self):
        """German umlauts should be replaced with ASCII equivalents."""
        assert "ue" in preprocess_sentence("über")
        assert "ae" in preprocess_sentence("Bär")
        assert "oe" in preprocess_sentence("schön")
        assert "ss" in preprocess_sentence("groß")

    def test_punctuation_spacing(self):
        """Punctuation should have spaces added around it."""
        result = preprocess_sentence("hello, world!")
        assert " , " in result
        assert " ! " in result or result.endswith(" !")

    def test_special_characters_removed(self):
        """Non-alphabetic characters should be removed."""
        result = preprocess_sentence("hello123world")
        assert "1" not in result
        assert "2" not in result
        assert "3" not in result

    def test_full_sentence(self):
        """Test a complete sentence preprocessing."""
        result = preprocess_sentence("Hello, how are you?")
        assert result == "hello , how are you ?"


class TestCreateTokenizer:
    """Tests for create_tokenizer function."""

    def test_tokenizer_creation(self):
        """Tokenizer should be created successfully."""
        sentences = ["hello world", "how are you"]
        tokenizer = create_tokenizer(sentences)
        assert tokenizer is not None

    def test_tokenizer_word_index(self):
        """Tokenizer should create word index for all words."""
        sentences = ["<start> hello world <end>", "<start> goodbye <end>"]
        tokenizer = create_tokenizer(sentences)

        assert "<start>" in tokenizer.word_index
        assert "<end>" in tokenizer.word_index
        assert "hello" in tokenizer.word_index

    def test_start_token_index(self):
        """<start> token should typically get index 1."""
        sentences = ["<start> hello <end>"] * 10
        tokenizer = create_tokenizer(sentences)
        # <start> should be the most common word and get index 1
        assert tokenizer.word_index.get("<start>") == 1


class TestTokenizeAndPad:
    """Tests for tokenize_and_pad function."""

    def test_output_shape(self):
        """Output should have correct shape."""
        sentences = ["<start> hello world <end>", "<start> hi <end>"]
        tokenizer = create_tokenizer(sentences)
        result = tokenize_and_pad(sentences, tokenizer, max_length=10)

        assert result.shape == (2, 10)

    def test_padding(self):
        """Shorter sequences should be padded with zeros."""
        sentences = ["<start> hi <end>"]
        tokenizer = create_tokenizer(sentences)
        result = tokenize_and_pad(sentences, tokenizer, max_length=10)

        # Check that there are zeros at the end (padding)
        assert result[0, -1] == 0

    def test_dtype(self):
        """Output should be a numpy array."""
        sentences = ["<start> hello <end>"]
        tokenizer = create_tokenizer(sentences)
        result = tokenize_and_pad(sentences, tokenizer, max_length=5)

        assert isinstance(result, np.ndarray)


class TestGetVocabSize:
    """Tests for get_vocab_size function."""

    def test_vocab_size(self):
        """Should return correct vocabulary size."""
        sentences = ["<start> hello world <end>", "<start> goodbye world <end>"]
        tokenizer = create_tokenizer(sentences)
        vocab_size = get_vocab_size(tokenizer)

        # Should have: <start>, <end>, hello, world, goodbye = 5 unique tokens
        assert vocab_size == 5

    def test_vocab_size_duplicates(self):
        """Duplicate words should not increase vocab size."""
        sentences = ["hello hello hello"]
        tokenizer = create_tokenizer(sentences)
        vocab_size = get_vocab_size(tokenizer)

        assert vocab_size == 1
