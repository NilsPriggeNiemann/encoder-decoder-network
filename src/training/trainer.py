"""Training utilities for the encoder-decoder translation model with GPU optimization."""

import logging
from pathlib import Path
from typing import Tuple, Optional

import tensorflow as tf

from src.config import MAX_GERMAN, LEARNING_RATE, EPOCHS, CHECKPOINT_DIR
from src.models.encoder import Encoder
from src.models.decoder import Decoder

logger = logging.getLogger(__name__)


def enable_mixed_precision() -> bool:
    """Enable mixed precision training for faster GPU computation.

    Mixed precision uses float16 for computations and float32 for
    variables, providing up to 3x speedup on modern GPUs (RTX 30/40/50 series).

    Returns:
        True if mixed precision was enabled, False otherwise.
    """
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        logger.info(f"Mixed precision enabled: {policy.name}")
        logger.info(f"Compute dtype: {policy.compute_dtype}")
        logger.info(f"Variable dtype: {policy.variable_dtype}")
        return True
    except Exception as e:
        logger.warning(f"Could not enable mixed precision: {e}")
        return False


def input_output(
    german_batch: tf.Tensor,
    vocab_size: int,
    max_length: int = MAX_GERMAN
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Prepare decoder inputs and targets from German sequences.

    For teacher forcing, the decoder input is the sequence without the last token,
    and the target is the sequence without the first token (one-hot encoded).

    Args:
        german_batch: Batch of German token sequences (batch_size, max_length).
        vocab_size: Size of the German vocabulary.
        max_length: Maximum sequence length.

    Returns:
        Tuple of:
            - decoder_input: Input sequences (batch_size, max_length - 1)
            - decoder_target: One-hot encoded targets
              (batch_size, max_length - 1, vocab_size + 1)
    """
    decoder_input = german_batch[:, :max_length - 1]
    decoder_target = tf.keras.utils.to_categorical(
        german_batch[:, 1:max_length],
        num_classes=vocab_size + 1
    )
    return decoder_input, decoder_target


def compute_loss(
    y_true: tf.Tensor,
    y_pred: tf.Tensor
) -> tf.Tensor:
    """Compute categorical cross-entropy loss.

    Args:
        y_true: Ground truth one-hot encoded labels.
        y_pred: Model predictions (logits).

    Returns:
        Scalar loss value.
    """
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    return tf.reduce_sum(loss_fn(y_true, y_pred))


@tf.function(jit_compile=True)
def train_step(
    encoder: Encoder,
    decoder: Decoder,
    optimizer: tf.keras.optimizers.Optimizer,
    english_batch: tf.Tensor,
    german_input: tf.Tensor,
    german_target: tf.Tensor
) -> tf.Tensor:
    """Perform a single training step with XLA compilation for GPU optimization.

    Args:
        encoder: Encoder model.
        decoder: Decoder model.
        optimizer: Optimizer instance.
        english_batch: Batch of embedded English sentences.
        german_input: Batch of German decoder inputs.
        german_target: Batch of German targets (one-hot encoded).

    Returns:
        Batch loss value.
    """
    with tf.GradientTape() as tape:
        # Forward pass through encoder
        _, hidden_state, cell_state = encoder(english_batch, training=True)

        # Forward pass through decoder
        decoder_output, _, _ = decoder(
            german_input,
            hidden_state=hidden_state,
            cell_state=cell_state,
            training=True
        )

        # Compute loss
        loss = compute_loss(german_target, decoder_output)

    # Compute and apply gradients
    trainable_vars = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, trainable_vars)
    optimizer.apply_gradients(zip(gradients, trainable_vars))

    return loss


@tf.function
def evaluate_step(
    encoder: Encoder,
    decoder: Decoder,
    english_batch: tf.Tensor,
    german_input: tf.Tensor,
    german_target: tf.Tensor
) -> tf.Tensor:
    """Perform a single evaluation step (no gradient computation).

    Args:
        encoder: Encoder model.
        decoder: Decoder model.
        english_batch: Batch of embedded English sentences.
        german_input: Batch of German decoder inputs.
        german_target: Batch of German targets (one-hot encoded).

    Returns:
        Batch loss value.
    """
    _, hidden_state, cell_state = encoder(english_batch, training=False)
    decoder_output, _, _ = decoder(
        german_input,
        hidden_state=hidden_state,
        cell_state=cell_state,
        training=False
    )
    return compute_loss(german_target, decoder_output)


def train(
    encoder: Encoder,
    decoder: Decoder,
    train_dataset: tf.data.Dataset,
    val_dataset: tf.data.Dataset,
    vocab_size: int,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    checkpoint_dir: Path = CHECKPOINT_DIR,
    val_batches: int = 50,
    use_mixed_precision: bool = True,
    use_xla: bool = True
) -> Tuple[list, list]:
    """Train the encoder-decoder model with GPU optimizations.

    Args:
        encoder: Encoder model.
        decoder: Decoder model.
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        vocab_size: Size of the German vocabulary.
        epochs: Number of training epochs.
        learning_rate: Learning rate for the optimizer.
        checkpoint_dir: Directory to save model checkpoints.
        val_batches: Number of validation batches to evaluate per epoch.
        use_mixed_precision: Enable mixed precision (float16) for faster training.
        use_xla: Enable XLA compilation for GPU optimization.

    Returns:
        Tuple of (train_losses, val_losses) lists.
    """
    # Enable mixed precision for faster GPU training
    if use_mixed_precision:
        enable_mixed_precision()

    # Enable XLA compilation
    if use_xla:
        tf.config.optimizer.set_jit(True)
        logger.info("XLA JIT compilation enabled")

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        epoch_loss = tf.keras.metrics.Mean()
        val_loss = tf.keras.metrics.Mean()

        # Training loop
        for english_batch, german_batch in train_dataset:
            german_input, german_target = input_output(german_batch, vocab_size)
            batch_loss = train_step(
                encoder, decoder, optimizer,
                english_batch, german_input, german_target
            )
            epoch_loss.update_state(batch_loss)

        # Save checkpoints
        encoder.save_weights(checkpoint_dir / 'encoder.weights.h5')
        decoder.save_weights(checkpoint_dir / 'decoder.weights.h5')

        # Validation loop
        for english_batch, german_batch in val_dataset.shuffle(100).take(val_batches):
            german_input, german_target = input_output(german_batch, vocab_size)
            batch_loss = evaluate_step(
                encoder, decoder,
                english_batch, german_input, german_target
            )
            val_loss.update_state(batch_loss)

        train_losses.append(float(epoch_loss.result()))
        val_losses.append(float(val_loss.result()))

        logger.info(
            f"Epoch {epoch + 1:03d} -- "
            f"Loss: {epoch_loss.result():.3f} -- "
            f"Validation Loss: {val_loss.result():.3f}"
        )

    return train_losses, val_losses
