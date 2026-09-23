"""Translation inference module for English to German translation."""

from pathlib import Path
from typing import Optional

import tensorflow as tf
import tensorflow_hub as hub

from src.config import (
    MAX_ENGLISH,
    MAX_GERMAN,
    START_TOKEN_ID,
    END_TOKEN_ID,
    CHECKPOINT_DIR,
)
from src.data.preprocessing import preprocess_sentence
from src.data.dataset import load_embedding_model
from src.models.encoder import Encoder
from src.models.decoder import Decoder


class Translator:
    """Translator class for English to German translation.

    This class handles the complete translation pipeline:
    1. Preprocess English input
    2. Embed English words
    3. Encode the sequence
    4. Decode German output using greedy decoding
    """

    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        tokenizer: tf.keras.preprocessing.text.Tokenizer,
        embedding_layer: hub.KerasLayer,
        max_english: int = MAX_ENGLISH,
        max_german: int = MAX_GERMAN
    ):
        """Initialize the translator.

        Args:
            encoder: Trained encoder model.
            decoder: Trained decoder model.
            tokenizer: Fitted German tokenizer.
            embedding_layer: Pre-trained English embedding layer.
            max_english: Maximum length of English sequences.
            max_german: Maximum length of German output.
        """
        self.encoder = encoder
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.embedding_layer = embedding_layer
        self.max_english = max_english
        self.max_german = max_german

    def _embed_sentence(self, sentence: str) -> tf.Tensor:
        """Preprocess and embed an English sentence.

        Args:
            sentence: Raw English sentence.

        Returns:
            Tensor of shape (1, max_english, embedding_dim).
        """
        # Preprocess
        processed = preprocess_sentence(sentence)

        # Split and embed
        tokens = tf.strings.split(processed)
        embeddings = self.embedding_layer(tokens)

        # Pad to max_english length
        num_words = tf.shape(embeddings)[0]
        padding = self.max_english - num_words

        # Create padding configuration
        pad_config = tf.constant([[0, 0], [0, 0]])
        if padding > 0:
            pad_config = tf.stack([
                tf.stack([tf.constant(0), padding]),
                tf.constant([0, 0])
            ])

        padded = tf.pad(embeddings, pad_config, "CONSTANT")

        # Ensure correct shape and add batch dimension
        padded = padded[:self.max_english, :]
        return tf.expand_dims(padded, axis=0)

    def translate(self, sentence: str) -> str:
        """Translate an English sentence to German.

        Uses greedy decoding: at each step, selects the word with
        the highest probability.

        Args:
            sentence: English sentence to translate.

        Returns:
            Translated German sentence.
        """
        # Embed the English sentence
        english_input = self._embed_sentence(sentence)

        # Encode
        _, hidden_state, cell_state = self.encoder(english_input, training=False)

        # Start with the <start> token
        german_sequence = tf.constant([[START_TOKEN_ID]], dtype=tf.int32)

        # Decode iteratively
        for _ in range(self.max_german):
            # Pad the German sequence
            current_length = german_sequence.shape[1]
            pad_length = self.max_german - current_length

            if pad_length > 0:
                padding = tf.zeros([1, pad_length], dtype=tf.int32)
                german_input = tf.concat([german_sequence, padding], axis=1)
            else:
                german_input = german_sequence[:, :self.max_german]

            # Get decoder output
            decoder_output, _, _ = self.decoder(
                german_input,
                hidden_state=hidden_state,
                cell_state=cell_state,
                training=False
            )

            # Get the prediction for the current position
            current_pos = german_sequence.shape[1] - 1
            next_word_logits = decoder_output[0, current_pos, :]
            next_word_id = tf.argmax(next_word_logits, axis=-1).numpy()

            # Stop if we predict the end token
            if next_word_id == END_TOKEN_ID:
                break

            # Append the predicted word
            german_sequence = tf.concat(
                [german_sequence, tf.constant([[next_word_id]], dtype=tf.int32)],
                axis=1
            )

        # Convert token IDs to text
        translation = self.tokenizer.sequences_to_texts(
            german_sequence.numpy().tolist()
        )[0]

        # Remove the <start> token from the output
        if translation.startswith('<start> '):
            translation = translation[8:]

        return translation.strip()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: Path,
        tokenizer: tf.keras.preprocessing.text.Tokenizer,
        vocab_size: int,
        embedding_layer: Optional[hub.KerasLayer] = None
    ) -> 'Translator':
        """Load a translator from saved checkpoints.

        Args:
            checkpoint_dir: Directory containing encoder.weights.h5 and
                            decoder.weights.h5 files.
            tokenizer: Fitted German tokenizer.
            vocab_size: Size of the German vocabulary.
            embedding_layer: Pre-trained English embedding layer.
                            If None, loads from TensorFlow Hub.

        Returns:
            Initialized Translator instance.
        """
        checkpoint_dir = Path(checkpoint_dir)

        # Load embedding layer if not provided
        if embedding_layer is None:
            embedding_layer = load_embedding_model()

        # Create and initialize models
        encoder = Encoder()
        decoder = Decoder(vocab_size=vocab_size)

        # Build models by calling them once
        dummy_english = tf.zeros([1, MAX_ENGLISH, 128])
        dummy_german = tf.zeros([1, MAX_GERMAN], dtype=tf.int32)

        encoder(dummy_english)
        decoder(dummy_german, hidden_state=tf.zeros([1, 512]), cell_state=tf.zeros([1, 512]))

        # Load weights
        encoder.load_weights(checkpoint_dir / 'encoder.weights.h5')
        decoder.load_weights(checkpoint_dir / 'decoder.weights.h5')

        return cls(
            encoder=encoder,
            decoder=decoder,
            tokenizer=tokenizer,
            embedding_layer=embedding_layer
        )
