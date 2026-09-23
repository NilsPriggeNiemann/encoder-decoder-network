"""Custom layers for the encoder-decoder model."""

import tensorflow as tf
from tensorflow.keras.layers import Layer

from src.config import EMBEDDING_DIM


class CustomEndTokenLayer(Layer):
    """Custom layer that adds a learnable end token to encoder inputs.

    This layer concatenates a trainable end-of-sequence token embedding
    to the end of each input sequence. The token is learned during training.

    Input shape: (batch_size, sequence_length, embedding_dim)
    Output shape: (batch_size, sequence_length + 1, embedding_dim)
    """

    def __init__(self, embedding_dim: int = EMBEDDING_DIM, **kwargs):
        """Initialize the custom end token layer.

        Args:
            embedding_dim: Dimension of the embedding space.
            **kwargs: Additional arguments passed to the base Layer class.
        """
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.end_token = None

    def build(self, input_shape):
        """Create the trainable end token weight."""
        self.end_token = self.add_weight(
            name='end_token',
            shape=[1, 1, self.embedding_dim],
            initializer='random_normal',
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Add the end token to each sequence in the batch.

        Args:
            inputs: Tensor of shape (batch_size, sequence_length, embedding_dim).

        Returns:
            Tensor of shape (batch_size, sequence_length + 1, embedding_dim).
        """
        batch_size = tf.shape(inputs)[0]
        tiling = tf.stack([batch_size, 1, 1])
        repeated_end_token = tf.tile(self.end_token, tiling)
        return tf.concat([inputs, repeated_end_token], axis=1)

    def get_config(self):
        """Return layer configuration for serialization."""
        config = super().get_config()
        config.update({'embedding_dim': self.embedding_dim})
        return config
