#!/usr/bin/env python
"""Gradio web interface for English to German translation."""

import pickle
from pathlib import Path

# Configure GPU before importing TensorFlow models
from src.utils.gpu import configure_gpu
configure_gpu(memory_growth=True)

import gradio as gr

from src.config import CHECKPOINT_DIR
from src.inference.translator import Translator


def load_translator(checkpoint_dir: Path = CHECKPOINT_DIR) -> Translator:
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


# Load translator at startup
print("Loading translation model...")
translator = load_translator()
print("Model loaded successfully!")


def translate(english_text: str) -> str:
    """Translate English text to German.

    Args:
        english_text: English sentence to translate.

    Returns:
        German translation.
    """
    if not english_text.strip():
        return ""

    return translator.translate(english_text)


# Example sentences for the demo
EXAMPLES = [
    ["Hello, how are you?"],
    ["What is your name?"],
    ["I love learning new languages."],
    ["Where is the train station?"],
    ["Thank you very much."],
    ["The weather is nice today."],
    ["Can you help me please?"],
    ["I would like a cup of coffee."],
]


# Create Gradio interface
demo = gr.Interface(
    fn=translate,
    inputs=gr.Textbox(
        label="English",
        placeholder="Enter an English sentence...",
        lines=2
    ),
    outputs=gr.Textbox(
        label="German",
        lines=2
    ),
    title="English to German Translator",
    description=(
        "A neural machine translation model using an encoder-decoder architecture "
        "with LSTM layers. This model was trained on the Tatoeba dataset."
    ),
    examples=EXAMPLES,
    cache_examples=False,
    theme=gr.themes.Soft()
)


if __name__ == "__main__":
    demo.launch()
