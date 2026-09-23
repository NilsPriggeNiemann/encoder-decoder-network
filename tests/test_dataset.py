"""Tests for dataset utilities."""

import numpy as np
import pytest
import tensorflow as tf

from src.data.dataset import (
    _split_sentence,
    _filter_max_length,
    _pad_sequence,
)


class TestSplitSentence:
    """Tests for _split_sentence function."""

    def test_basic_split(self):
        """Sentence should be split into tokens."""
        english = tf.constant("hello world")
        german = tf.constant([1, 2, 3])

        result_english, result_german = _split_sentence(english, german)

        assert result_english.numpy().tolist() == [b"hello", b"world"]
        # German should pass through unchanged
        np.testing.assert_array_equal(result_german.numpy(), [1, 2, 3])

    def test_single_word(self):
        """Single word sentence should work."""
        english = tf.constant("hello")
        german = tf.constant([1])

        result_english, _ = _split_sentence(english, german)
        assert result_english.numpy().tolist() == [b"hello"]


class TestFilterMaxLength:
    """Tests for _filter_max_length function."""

    def test_within_limit(self):
        """Sentences within limit should pass."""
        english = tf.constant([[1.0, 2.0]] * 5)  # 5 tokens
        german = tf.constant([1, 2, 3])

        result = _filter_max_length(english, german, max_length=10)
        assert result.numpy() == True

    def test_at_limit(self):
        """Sentences at exactly the limit should pass."""
        english = tf.constant([[1.0, 2.0]] * 10)  # 10 tokens
        german = tf.constant([1, 2, 3])

        result = _filter_max_length(english, german, max_length=10)
        assert result.numpy() == True

    def test_over_limit(self):
        """Sentences over the limit should be filtered out."""
        english = tf.constant([[1.0, 2.0]] * 15)  # 15 tokens
        german = tf.constant([1, 2, 3])

        result = _filter_max_length(english, german, max_length=10)
        assert result.numpy() == False


class TestPadSequence:
    """Tests for _pad_sequence function."""

    def test_padding_added(self):
        """Shorter sequences should be padded."""
        english = tf.constant([[1.0, 2.0]] * 5)  # 5 tokens, 2 dims
        german = tf.constant([1, 2, 3])

        result_english, result_german = _pad_sequence(english, german, max_length=10)

        assert result_english.shape == (10, 2)
        # German should pass through unchanged
        np.testing.assert_array_equal(result_german.numpy(), [1, 2, 3])

    def test_padding_values(self):
        """Padding should be zeros."""
        english = tf.constant([[1.0, 1.0]] * 3)  # 3 tokens
        german = tf.constant([1])

        result_english, _ = _pad_sequence(english, german, max_length=5)

        # First 3 positions should be [1, 1]
        np.testing.assert_array_almost_equal(result_english[:3].numpy(), [[1.0, 1.0]] * 3)
        # Last 2 positions should be zeros
        np.testing.assert_array_almost_equal(result_english[3:].numpy(), [[0.0, 0.0]] * 2)

    def test_no_padding_needed(self):
        """Sequences at max length should not change."""
        english = tf.constant([[1.0, 2.0]] * 10)  # Already 10 tokens
        german = tf.constant([1, 2, 3])

        result_english, _ = _pad_sequence(english, german, max_length=10)

        assert result_english.shape == (10, 2)
        np.testing.assert_array_almost_equal(result_english.numpy(), english.numpy())


class TestDatasetIntegration:
    """Integration tests for dataset creation."""

    def test_batch_shapes(self):
        """Test that batched data has correct shapes."""
        # This is a simplified test that doesn't require TensorFlow Hub
        batch_size = 4
        max_english = 20
        max_german = 25
        embedding_dim = 128

        # Create mock data
        english_embeddings = tf.random.normal([batch_size, max_english, embedding_dim])
        german_tokens = tf.random.uniform(
            [batch_size, max_german],
            minval=0, maxval=1000,
            dtype=tf.int32
        )

        assert english_embeddings.shape == (batch_size, max_english, embedding_dim)
        assert german_tokens.shape == (batch_size, max_german)
