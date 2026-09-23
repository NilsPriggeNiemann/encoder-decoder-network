"""Decoder model for sequence-to-sequence translation."""

from typing import Tuple, Optional

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense

from src.config import EMBEDDING_DIM, LSTM_UNITS


class Decoder(Model):
    """LSTM-based decoder for the translation model.

    Architecture:
    1. Embedding layer (German vocabulary, masks zeros)
    2. LSTM layer (initialized with encoder states)
    3. Dense layer (outputs logits for vocabulary)

    Input: German token IDs + encoder hidden/cell states
    Output: Logits for each position in the sequence
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = EMBEDDING_DIM,
        lstm_units: int = LSTM_UNITS,
        **kwargs
    ):
        """Initialize the decoder model.

        Args:
            vocab_size: Size of the German vocabulary.
            embedding_dim: Dimension of the embedding space.
            lstm_units: Number of units in the LSTM layer.
            **kwargs: Additional arguments passed to the base Model class.
        """
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units

        # Add 1 to vocab_size for padding token (index 0)
        self.embedding = Embedding(
            vocab_size + 1,
            embedding_dim,
            mask_zero=True
        )
        self.lstm = LSTM(
            lstm_units,
            return_sequences=True,
            return_state=True
        )
        self.dense = Dense(vocab_size + 1)

    def call(
        self,
        inputs: tf.Tensor,
        hidden_state: Optional[tf.Tensor] = None,
        cell_state: Optional[tf.Tensor] = None,
        training: bool = None
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Forward pass through the decoder.

        Args:
            inputs: German token IDs of shape (batch_size, sequence_length).
            hidden_state: Initial hidden state from encoder (batch_size, lstm_units).
            cell_state: Initial cell state from encoder (batch_size, lstm_units).
            training: Whether the model is in training mode.

        Returns:
            Tuple of:
                - logits: Output logits (batch_size, sequence_length, vocab_size + 1)
                - hidden_state: Final hidden state (batch_size, lstm_units)
                - cell_state: Final cell state (batch_size, lstm_units)
        """
        x = self.embedding(inputs)

        initial_state = None
        if hidden_state is not None and cell_state is not None:
            initial_state = [hidden_state, cell_state]

        x, hidden_state, cell_state = self.lstm(x, initial_state=initial_state)
        logits = self.dense(x)

        return logits, hidden_state, cell_state

    def get_config(self):
        """Return model configuration for serialization."""
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'embedding_dim': self.embedding_dim,
            'lstm_units': self.lstm_units,
        })
        return config
