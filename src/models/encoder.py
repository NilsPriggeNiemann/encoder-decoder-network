"""Encoder model for sequence-to-sequence translation."""

from typing import Tuple

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Masking

from src.config import EMBEDDING_DIM, LSTM_UNITS
from src.models.layers import CustomEndTokenLayer


class Encoder(Model):
    """LSTM-based encoder for the translation model.

    Architecture:
    1. Custom end token layer (adds learnable end-of-sequence token)
    2. Masking layer (masks zero-padded positions)
    3. LSTM layer (returns sequences and hidden/cell states)

    Input shape: (batch_size, sequence_length, embedding_dim)
    Output: Tuple of (sequence_output, hidden_state, cell_state)
    """

    def __init__(
        self,
        lstm_units: int = LSTM_UNITS,
        embedding_dim: int = EMBEDDING_DIM,
        **kwargs
    ):
        """Initialize the encoder model.

        Args:
            lstm_units: Number of units in the LSTM layer.
            embedding_dim: Dimension of input embeddings.
            **kwargs: Additional arguments passed to the base Model class.
        """
        super().__init__(**kwargs)
        self.lstm_units = lstm_units
        self.embedding_dim = embedding_dim

        self.custom_layer = CustomEndTokenLayer(embedding_dim=embedding_dim)
        self.masking_layer = Masking(
            mask_value=tf.zeros([1, embedding_dim], dtype=tf.float32)
        )
        self.lstm_layer = LSTM(
            lstm_units,
            return_sequences=True,
            return_state=True
        )

    def call(
        self,
        inputs: tf.Tensor,
        training: bool = None
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Forward pass through the encoder.

        Args:
            inputs: English sentence embeddings of shape
                    (batch_size, sequence_length, embedding_dim).
            training: Whether the model is in training mode.

        Returns:
            Tuple of:
                - sequence_output: LSTM outputs for all timesteps
                  (batch_size, sequence_length + 1, lstm_units)
                - hidden_state: Final hidden state (batch_size, lstm_units)
                - cell_state: Final cell state (batch_size, lstm_units)
        """
        x = self.custom_layer(inputs)
        x = self.masking_layer(x)
        sequence_output, hidden_state, cell_state = self.lstm_layer(x)
        return sequence_output, hidden_state, cell_state

    def get_config(self):
        """Return model configuration for serialization."""
        config = super().get_config()
        config.update({
            'lstm_units': self.lstm_units,
            'embedding_dim': self.embedding_dim,
        })
        return config
