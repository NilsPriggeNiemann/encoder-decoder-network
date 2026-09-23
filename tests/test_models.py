"""Tests for neural network model components."""

import pytest
import tensorflow as tf

from src.models.layers import CustomEndTokenLayer
from src.models.encoder import Encoder
from src.models.decoder import Decoder
from src.config import EMBEDDING_DIM, LSTM_UNITS


class TestCustomEndTokenLayer:
    """Tests for CustomEndTokenLayer."""

    def test_output_shape(self):
        """Output should have sequence length + 1."""
        layer = CustomEndTokenLayer(embedding_dim=128)
        inputs = tf.random.normal([4, 10, 128])  # batch=4, seq=10, emb=128
        outputs = layer(inputs)

        assert outputs.shape == (4, 11, 128)

    def test_trainable_weight(self):
        """Layer should have a trainable end token weight."""
        layer = CustomEndTokenLayer(embedding_dim=64)
        _ = layer(tf.random.normal([2, 5, 64]))

        assert len(layer.trainable_weights) == 1
        assert layer.trainable_weights[0].shape == (1, 1, 64)

    def test_different_batch_sizes(self):
        """Layer should work with different batch sizes."""
        layer = CustomEndTokenLayer(embedding_dim=128)

        for batch_size in [1, 4, 16, 32]:
            inputs = tf.random.normal([batch_size, 10, 128])
            outputs = layer(inputs)
            assert outputs.shape[0] == batch_size

    def test_config_serialization(self):
        """Layer config should be serializable."""
        layer = CustomEndTokenLayer(embedding_dim=64)
        config = layer.get_config()

        assert 'embedding_dim' in config
        assert config['embedding_dim'] == 64


class TestEncoder:
    """Tests for Encoder model."""

    def test_output_shapes(self):
        """Encoder should return correct output shapes."""
        encoder = Encoder(lstm_units=256, embedding_dim=128)
        inputs = tf.random.normal([4, 20, 128])

        sequence_output, hidden_state, cell_state = encoder(inputs)

        # Sequence output: (batch, seq+1, units) - +1 from end token
        assert sequence_output.shape == (4, 21, 256)
        # Hidden state: (batch, units)
        assert hidden_state.shape == (4, 256)
        # Cell state: (batch, units)
        assert cell_state.shape == (4, 256)

    def test_default_config(self):
        """Encoder with default config should work."""
        encoder = Encoder()
        inputs = tf.random.normal([2, 20, EMBEDDING_DIM])

        outputs = encoder(inputs)
        assert len(outputs) == 3

    def test_masking(self):
        """Encoder should handle zero-padded inputs."""
        encoder = Encoder(lstm_units=128, embedding_dim=64)

        # Create input with some zeros (padding)
        inputs = tf.concat([
            tf.random.normal([2, 10, 64]),
            tf.zeros([2, 5, 64])
        ], axis=1)

        # Should not raise an error
        outputs = encoder(inputs)
        assert outputs[0].shape == (2, 16, 128)


class TestDecoder:
    """Tests for Decoder model."""

    def test_output_shapes(self):
        """Decoder should return correct output shapes."""
        vocab_size = 1000
        decoder = Decoder(vocab_size=vocab_size, lstm_units=256, embedding_dim=128)

        inputs = tf.random.uniform([4, 25], minval=0, maxval=vocab_size, dtype=tf.int32)
        hidden = tf.random.normal([4, 256])
        cell = tf.random.normal([4, 256])

        logits, new_hidden, new_cell = decoder(inputs, hidden_state=hidden, cell_state=cell)

        # Logits: (batch, seq, vocab_size + 1)
        assert logits.shape == (4, 25, vocab_size + 1)
        # Hidden state: (batch, units)
        assert new_hidden.shape == (4, 256)
        # Cell state: (batch, units)
        assert new_cell.shape == (4, 256)

    def test_without_initial_state(self):
        """Decoder should work without initial state."""
        vocab_size = 500
        decoder = Decoder(vocab_size=vocab_size)

        inputs = tf.random.uniform([2, 10], minval=0, maxval=vocab_size, dtype=tf.int32)
        logits, hidden, cell = decoder(inputs)

        assert logits.shape == (2, 10, vocab_size + 1)

    def test_embedding_mask_zero(self):
        """Decoder should handle zero tokens (padding)."""
        vocab_size = 100
        decoder = Decoder(vocab_size=vocab_size, lstm_units=64)

        # Input with some zero padding
        inputs = tf.constant([[1, 5, 10, 0, 0], [1, 3, 0, 0, 0]], dtype=tf.int32)

        # Should not raise an error
        logits, _, _ = decoder(inputs)
        assert logits.shape == (2, 5, vocab_size + 1)


class TestEncoderDecoderIntegration:
    """Integration tests for encoder-decoder pipeline."""

    def test_end_to_end_forward_pass(self):
        """Test complete forward pass through encoder and decoder."""
        vocab_size = 500
        batch_size = 4
        english_seq_len = 20
        german_seq_len = 25

        encoder = Encoder(lstm_units=256, embedding_dim=128)
        decoder = Decoder(vocab_size=vocab_size, lstm_units=256, embedding_dim=128)

        # English embeddings
        english_input = tf.random.normal([batch_size, english_seq_len, 128])

        # German token IDs
        german_input = tf.random.uniform(
            [batch_size, german_seq_len],
            minval=0, maxval=vocab_size,
            dtype=tf.int32
        )

        # Forward pass through encoder
        _, hidden_state, cell_state = encoder(english_input)

        # Forward pass through decoder
        logits, _, _ = decoder(
            german_input,
            hidden_state=hidden_state,
            cell_state=cell_state
        )

        # Verify output shape
        assert logits.shape == (batch_size, german_seq_len, vocab_size + 1)

    def test_gradients_flow(self):
        """Test that gradients flow through both models."""
        vocab_size = 100
        encoder = Encoder(lstm_units=64, embedding_dim=32)
        decoder = Decoder(vocab_size=vocab_size, lstm_units=64, embedding_dim=32)

        english_input = tf.random.normal([2, 10, 32])
        german_input = tf.random.uniform([2, 8], minval=1, maxval=vocab_size, dtype=tf.int32)
        german_target = tf.one_hot(
            tf.random.uniform([2, 8], minval=1, maxval=vocab_size, dtype=tf.int32),
            depth=vocab_size + 1
        )

        with tf.GradientTape() as tape:
            _, hidden, cell = encoder(english_input)
            logits, _, _ = decoder(german_input, hidden_state=hidden, cell_state=cell)

            loss = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(german_target, logits, from_logits=True)
            )

        trainable_vars = encoder.trainable_variables + decoder.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # All gradients should be non-None
        assert all(g is not None for g in gradients)
