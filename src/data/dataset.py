"""TensorFlow Dataset utilities for the translation model with GPU optimization."""

from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from src.config import EMBEDDING_URL, MAX_ENGLISH, BATCH_SIZE


def load_embedding_model() -> hub.KerasLayer:
    """Load the pre-trained English word embedding from TensorFlow Hub.

    Returns:
        KerasLayer that embeds English words into 128-dimensional space.
    """
    return hub.KerasLayer(
        EMBEDDING_URL,
        output_shape=[128],
        input_shape=[],
        dtype=tf.string
    )


def _split_sentence(
    english: tf.Tensor,
    german: tf.Tensor
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Split English sentence into tokens."""
    return tf.strings.split(english), german


def _embed_sentence(
    embedding_layer: hub.KerasLayer
):
    """Create a function that embeds English tokens."""
    def embed(english: tf.Tensor, german: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        return embedding_layer(english), german
    return embed


def _filter_max_length(
    english: tf.Tensor,
    german: tf.Tensor,
    max_length: int = MAX_ENGLISH
) -> tf.Tensor:
    """Filter out sentences longer than max_length."""
    return tf.shape(english)[0] <= max_length


def _pad_sequence(
    english: tf.Tensor,
    german: tf.Tensor,
    max_length: int = MAX_ENGLISH
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Pad English embeddings to max_length with zero vectors."""
    current_length = tf.shape(english)[0]
    padding_length = max_length - current_length

    # Create padding matrix: [[padding_length, 0], [0, 0]]
    pad_config = tf.concat([
        tf.expand_dims(
            tf.concat([
                tf.expand_dims(padding_length, axis=0),
                tf.constant([0], dtype=tf.int32)
            ], axis=0),
            axis=0
        ),
        tf.zeros([1, 2], dtype=tf.int32)
    ], axis=0)

    english_padded = tf.pad(english, pad_config, "CONSTANT")
    return english_padded, german


def create_dataset(
    english_sentences: list,
    german_tokenized: np.ndarray,
    embedding_layer: hub.KerasLayer,
    batch_size: int = BATCH_SIZE,
    max_english_length: int = MAX_ENGLISH,
    prefetch: bool = True,
    cache: bool = False
) -> tf.data.Dataset:
    """Create a tf.data.Dataset for training with GPU optimization.

    Args:
        english_sentences: List of preprocessed English sentences.
        german_tokenized: NumPy array of tokenized and padded German sequences.
        embedding_layer: Pre-trained embedding layer for English words.
        batch_size: Batch size for the dataset.
        max_english_length: Maximum length of English sequences.
        prefetch: If True, prefetch batches to GPU for faster training.
        cache: If True, cache the dataset in memory (use for smaller datasets).

    Returns:
        Batched tf.data.Dataset with (english_embeddings, german_tokens) pairs.
    """
    dataset = tf.data.Dataset.from_tensor_slices(
        (english_sentences, german_tokenized)
    )

    # Apply transformations with parallel processing
    dataset = dataset.map(
        _split_sentence,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.map(
        _embed_sentence(embedding_layer),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.filter(
        lambda x, y: _filter_max_length(x, y, max_english_length)
    )
    dataset = dataset.map(
        lambda x, y: _pad_sequence(x, y, max_english_length),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # Cache dataset in memory if requested (useful for smaller datasets)
    if cache:
        dataset = dataset.cache()

    # Batch the data
    dataset = dataset.batch(batch_size)

    # Prefetch to GPU for optimal performance
    if prefetch:
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def prepare_train_val_split(
    english_sentences: list,
    german_tokenized: np.ndarray,
    embedding_layer: hub.KerasLayer,
    train_ratio: float = 0.8,
    batch_size: int = BATCH_SIZE,
    max_english_length: int = MAX_ENGLISH,
    shuffle: bool = True,
    seed: int = 42,
    prefetch: bool = True
) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Split data into training and validation datasets with GPU optimization.

    Args:
        english_sentences: List of preprocessed English sentences.
        german_tokenized: NumPy array of tokenized and padded German sequences.
        embedding_layer: Pre-trained embedding layer for English words.
        train_ratio: Fraction of data to use for training.
        batch_size: Batch size for both datasets.
        max_english_length: Maximum length of English sequences.
        shuffle: Whether to shuffle before splitting.
        seed: Random seed for reproducibility.
        prefetch: If True, prefetch batches to GPU.

    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    n_samples = len(english_sentences)
    indices = np.arange(n_samples)

    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)

    split_idx = int(n_samples * train_ratio)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    # Split the data
    train_english = [english_sentences[i] for i in train_indices]
    train_german = german_tokenized[train_indices]

    val_english = [english_sentences[i] for i in val_indices]
    val_german = german_tokenized[val_indices]

    # Create datasets with GPU optimization
    train_dataset = create_dataset(
        train_english, train_german, embedding_layer,
        batch_size, max_english_length, prefetch=prefetch
    )
    val_dataset = create_dataset(
        val_english, val_german, embedding_layer,
        batch_size, max_english_length, prefetch=prefetch
    )

    return train_dataset, val_dataset
