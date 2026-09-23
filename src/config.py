"""Configuration constants for the encoder-decoder translation model."""

from pathlib import Path

# Data configuration
NUM_EXAMPLES: int = 200000

# Sequence length limits
MAX_GERMAN: int = 25
MAX_ENGLISH: int = 20

# Training configuration
BATCH_SIZE: int = 32
LEARNING_RATE: float = 0.001
EPOCHS: int = 15

# Model architecture
EMBEDDING_DIM: int = 128
LSTM_UNITS: int = 512

# Special tokens
START_TOKEN: str = "<start>"
END_TOKEN: str = "<end>"
START_TOKEN_ID: int = 1
END_TOKEN_ID: int = 2

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

# TensorFlow Hub embedding URL
EMBEDDING_URL: str = "https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1"
