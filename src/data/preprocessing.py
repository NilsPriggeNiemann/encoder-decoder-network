"""Text preprocessing utilities for English-German translation."""

import re
import unicodedata
from typing import Tuple, List

import numpy as np
import tensorflow as tf

from src.config import START_TOKEN, END_TOKEN, MAX_GERMAN


def unicode_to_ascii(text: str) -> str:
    """Convert unicode string to ASCII by removing diacritics.

    Args:
        text: Unicode string to convert.

    Returns:
        ASCII string with diacritics removed.
    """
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )


def preprocess_sentence(sentence: str) -> str:
    """Preprocess a sentence for the translation model.

    Applies the following transformations:
    - Lowercase and strip whitespace
    - Replace German umlauts with ASCII equivalents
    - Convert to ASCII
    - Add spaces around punctuation
    - Remove non-alphabetic characters (except punctuation)
    - Normalize whitespace

    Args:
        sentence: Raw sentence string.

    Returns:
        Preprocessed sentence string.
    """
    sentence = sentence.lower().strip()

    # Replace German umlauts
    sentence = re.sub(r"ü", 'ue', sentence)
    sentence = re.sub(r"ä", 'ae', sentence)
    sentence = re.sub(r"ö", 'oe', sentence)
    sentence = re.sub(r'ß', 'ss', sentence)

    sentence = unicode_to_ascii(sentence)

    # Add spaces around punctuation
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)

    # Keep only letters and basic punctuation
    sentence = re.sub(r"[^a-z?.!,']+", " ", sentence)

    # Normalize whitespace
    sentence = re.sub(r'[" "]+', " ", sentence)

    return sentence.strip()


def load_dataset(
    path: str,
    num_examples: int
) -> Tuple[List[str], List[str]]:
    """Load English-German sentence pairs from a tab-separated file.

    Args:
        path: Path to the data file (tab-separated, English\tGerman format).
        num_examples: Maximum number of examples to load.

    Returns:
        Tuple of (english_sentences, german_sentences) lists.
    """
    english_sentences = []
    german_sentences = []

    with open(path, 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            if i >= num_examples:
                break

            parts = line.strip().split('\t')
            if len(parts) >= 2:
                english = preprocess_sentence(parts[0])
                german = f"{START_TOKEN} {preprocess_sentence(parts[1])} {END_TOKEN}"

                english_sentences.append(english)
                german_sentences.append(german)

    return english_sentences, german_sentences


def create_tokenizer(
    sentences: List[str]
) -> tf.keras.preprocessing.text.Tokenizer:
    """Create and fit a tokenizer on the given sentences.

    Args:
        sentences: List of sentences to fit the tokenizer on.

    Returns:
        Fitted Keras Tokenizer.
    """
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    tokenizer.fit_on_texts(sentences)
    return tokenizer


def tokenize_and_pad(
    sentences: List[str],
    tokenizer: tf.keras.preprocessing.text.Tokenizer,
    max_length: int = MAX_GERMAN
) -> np.ndarray:
    """Tokenize sentences and pad to a fixed length.

    Args:
        sentences: List of sentences to tokenize.
        tokenizer: Fitted Keras Tokenizer.
        max_length: Maximum sequence length (sequences are post-padded).

    Returns:
        NumPy array of shape (num_sentences, max_length) with token IDs.
    """
    sequences = tokenizer.texts_to_sequences(sentences)
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        sequences,
        maxlen=max_length,
        padding='post'
    )
    return np.array(padded)


def get_vocab_size(tokenizer: tf.keras.preprocessing.text.Tokenizer) -> int:
    """Get the vocabulary size from a tokenizer.

    Args:
        tokenizer: Fitted Keras Tokenizer.

    Returns:
        Number of unique tokens in the vocabulary.
    """
    return len(tokenizer.word_index)
