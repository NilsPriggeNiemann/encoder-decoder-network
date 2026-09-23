#!/usr/bin/env python
"""CLI script for training the encoder-decoder translation model."""

import argparse
import logging
import pickle
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure GPU before importing TensorFlow models
from src.utils.gpu import configure_gpu, get_device_info

# Must configure GPU before other TensorFlow imports
configure_gpu(memory_growth=True)

import matplotlib.pyplot as plt

from src.config import (
    NUM_EXAMPLES,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    DATA_DIR,
    CHECKPOINT_DIR,
)
from src.data.preprocessing import (
    load_dataset,
    create_tokenizer,
    tokenize_and_pad,
    get_vocab_size,
)
from src.data.dataset import load_embedding_model, prepare_train_val_split
from src.models.encoder import Encoder
from src.models.decoder import Decoder
from src.training.trainer import train

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train the English-German translation model.'
    )
    parser.add_argument(
        '--data-path',
        type=Path,
        default=DATA_DIR / 'deu.txt',
        help='Path to the English-German data file.'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=Path,
        default=CHECKPOINT_DIR,
        help='Directory to save model checkpoints.'
    )
    parser.add_argument(
        '--num-examples',
        type=int,
        default=NUM_EXAMPLES,
        help='Number of training examples to use.'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=EPOCHS,
        help='Number of training epochs.'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=BATCH_SIZE,
        help='Training batch size.'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=LEARNING_RATE,
        help='Learning rate for the optimizer.'
    )
    parser.add_argument(
        '--plot-losses',
        action='store_true',
        help='Plot training and validation losses after training.'
    )
    return parser.parse_args()


def plot_losses(train_losses: list, val_losses: list, save_path: Path):
    """Plot and save training curves."""
    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')

    axes[0].set_ylabel("Training Loss", fontsize=14)
    axes[0].plot(train_losses)

    axes[1].set_ylabel("Validation Loss", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].plot(val_losses)

    plt.savefig(save_path)
    logger.info(f"Training curves saved to {save_path}")
    plt.show()


def main():
    """Main training function."""
    args = parse_args()

    # Log GPU information
    device_info = get_device_info()
    logger.info(f"TensorFlow version: {device_info['tensorflow_version']}")
    logger.info(f"CUDA available: {device_info['cuda_available']}")
    logger.info(f"GPUs available: {device_info['num_gpus']}")
    if device_info['num_gpus'] > 0:
        for gpu in device_info.get('gpu_details', []):
            logger.info(f"  GPU: {gpu['name']} - Memory: {gpu['memory_limit'] / 1e9:.2f} GB")

    # Create checkpoint directory
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading data from {args.data_path}")
    english_sentences, german_sentences = load_dataset(
        str(args.data_path),
        args.num_examples
    )
    logger.info(f"Loaded {len(english_sentences)} sentence pairs")

    # Create tokenizer for German
    logger.info("Creating German tokenizer")
    tokenizer = create_tokenizer(german_sentences)
    vocab_size = get_vocab_size(tokenizer)
    logger.info(f"German vocabulary size: {vocab_size}")

    # Save tokenizer for inference
    tokenizer_path = args.checkpoint_dir / 'tokenizer.pkl'
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    logger.info(f"Tokenizer saved to {tokenizer_path}")

    # Tokenize and pad German sentences
    german_tokenized = tokenize_and_pad(german_sentences, tokenizer)

    # Load embedding model
    logger.info("Loading English embedding model from TensorFlow Hub")
    embedding_layer = load_embedding_model()

    # Create datasets
    logger.info("Preparing training and validation datasets")
    train_dataset, val_dataset = prepare_train_val_split(
        english_sentences,
        german_tokenized,
        embedding_layer,
        batch_size=args.batch_size
    )

    # Create models
    logger.info("Creating encoder and decoder models")
    encoder = Encoder()
    decoder = Decoder(vocab_size=vocab_size)

    # Train
    logger.info(f"Starting training for {args.epochs} epochs")
    train_losses, val_losses = train(
        encoder,
        decoder,
        train_dataset,
        val_dataset,
        vocab_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir
    )

    logger.info("Training complete!")
    logger.info(f"Final training loss: {train_losses[-1]:.3f}")
    logger.info(f"Final validation loss: {val_losses[-1]:.3f}")

    # Save vocab size for inference
    with open(args.checkpoint_dir / 'vocab_size.txt', 'w') as f:
        f.write(str(vocab_size))

    if args.plot_losses:
        plot_losses(train_losses, val_losses, args.checkpoint_dir / 'losses.png')


if __name__ == '__main__':
    main()
