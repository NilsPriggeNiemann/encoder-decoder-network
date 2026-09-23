#!/usr/bin/env python
"""CLI script for translating English sentences to German."""

import argparse
import pickle
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Configure GPU before importing TensorFlow models
from src.utils.gpu import configure_gpu
configure_gpu(memory_growth=True)

from src.config import CHECKPOINT_DIR
from src.inference.translator import Translator


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Translate English sentences to German.'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=Path,
        default=CHECKPOINT_DIR,
        help='Directory containing model checkpoints.'
    )
    parser.add_argument(
        '--sentence',
        type=str,
        default=None,
        help='English sentence to translate. If not provided, enters interactive mode.'
    )
    return parser.parse_args()


def load_translator(checkpoint_dir: Path) -> Translator:
    """Load the translator from checkpoints.

    Args:
        checkpoint_dir: Directory containing model checkpoints.

    Returns:
        Initialized Translator instance.
    """
    # Load tokenizer
    tokenizer_path = checkpoint_dir / 'tokenizer.pkl'
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    # Load vocab size
    with open(checkpoint_dir / 'vocab_size.txt', 'r') as f:
        vocab_size = int(f.read().strip())

    return Translator.from_checkpoint(
        checkpoint_dir=checkpoint_dir,
        tokenizer=tokenizer,
        vocab_size=vocab_size
    )


def interactive_mode(translator: Translator):
    """Run the translator in interactive mode.

    Args:
        translator: Initialized Translator instance.
    """
    print("English to German Translator")
    print("Type 'quit' or 'exit' to stop.")
    print("-" * 40)

    while True:
        try:
            sentence = input("\nEnglish: ").strip()

            if sentence.lower() in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break

            if not sentence:
                continue

            translation = translator.translate(sentence)
            print(f"German:  {translation}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


def main():
    """Main translation function."""
    args = parse_args()

    print(f"Loading model from {args.checkpoint_dir}")
    translator = load_translator(args.checkpoint_dir)
    print("Model loaded successfully!")

    if args.sentence:
        # Single sentence mode
        translation = translator.translate(args.sentence)
        print(f"English: {args.sentence}")
        print(f"German:  {translation}")
    else:
        # Interactive mode
        interactive_mode(translator)


if __name__ == '__main__':
    main()
