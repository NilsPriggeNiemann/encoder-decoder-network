# Encoder-Decoder Translation Network

A neural machine translation model that translates English sentences to German using an encoder-decoder architecture with LSTM layers.

## Overview

This project implements a sequence-to-sequence model for English-to-German translation. It was developed as part of a TensorFlow specialization course on Coursera and demonstrates:

- Custom Keras layers and model subclassing
- Pre-trained word embeddings from TensorFlow Hub
- Custom training loops with gradient tape
- Teacher forcing during training
- Greedy decoding during inference

## Model Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         ENCODER                              │
├─────────────────────────────────────────────────────────────┤
│  English Words → Pre-trained Embedding (128-dim, frozen)     │
│        ↓                                                     │
│  Custom End Token Layer (learnable)                          │
│        ↓                                                     │
│  Masking Layer                                               │
│        ↓                                                     │
│  LSTM (512 units) → [hidden_state, cell_state]              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                         DECODER                              │
├─────────────────────────────────────────────────────────────┤
│  German Tokens → Embedding (128-dim, learnable)              │
│        ↓                                                     │
│  LSTM (512 units, initialized with encoder states)           │
│        ↓                                                     │
│  Dense Layer → Logits (vocab_size)                          │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
encoder-decoder-network/
├── src/
│   ├── config.py              # Configuration constants
│   ├── data/
│   │   ├── preprocessing.py   # Text preprocessing
│   │   └── dataset.py         # TensorFlow dataset utilities
│   ├── models/
│   │   ├── layers.py          # Custom end token layer
│   │   ├── encoder.py         # Encoder model
│   │   └── decoder.py         # Decoder model
│   ├── training/
│   │   └── trainer.py         # Training loop
│   └── inference/
│       └── translator.py      # Translation inference
├── scripts/
│   ├── train.py               # CLI for training
│   └── translate.py           # CLI for inference
├── app.py                     # Gradio web demo
├── tests/                     # Unit tests
├── notebooks/                 # Original Jupyter notebook
├── requirements.txt
└── pyproject.toml
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/encoder-decoder-network.git
cd encoder-decoder-network
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Download the training data from [Tatoeba](http://www.manythings.org/anki/) and place `deu.txt` in the `data/` directory.

## Usage

### Training

Train the model from scratch:

```bash
python scripts/train.py --data-path data/deu.txt --epochs 15 --batch-size 32
```

Options:
- `--data-path`: Path to the English-German data file
- `--checkpoint-dir`: Directory to save model weights (default: `checkpoints/`)
- `--num-examples`: Number of training examples (default: 200000)
- `--epochs`: Number of training epochs (default: 15)
- `--batch-size`: Training batch size (default: 32)
- `--learning-rate`: Learning rate (default: 0.001)
- `--plot-losses`: Plot training curves after training

### Translation (CLI)

Translate a single sentence:

```bash
python scripts/translate.py --sentence "Hello, how are you?"
```

Or run in interactive mode:

```bash
python scripts/translate.py
```

### Web Demo (Gradio)

Launch the Gradio web interface:

```bash
python app.py
```

Then open http://localhost:7860 in your browser.

## Running Tests

```bash
pytest tests/ -v
```

## Configuration

Key hyperparameters can be modified in `src/config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NUM_EXAMPLES` | 200000 | Training examples to use |
| `MAX_ENGLISH` | 20 | Max English sentence length |
| `MAX_GERMAN` | 25 | Max German sentence length |
| `BATCH_SIZE` | 32 | Training batch size |
| `LSTM_UNITS` | 512 | LSTM hidden units |
| `EMBEDDING_DIM` | 128 | Embedding dimension |
| `LEARNING_RATE` | 0.001 | Adam learning rate |
| `EPOCHS` | 15 | Training epochs |

## Example Translations

```
English: Hello, how are you?
German:  hallo , wie geht es dir ?

English: What is your name?
German:  wie heisst du ?

English: I love learning new languages.
German:  ich lerne gern neue sprachen .
```

## Limitations

- Uses greedy decoding (no beam search)
- No attention mechanism
- Limited vocabulary size
- Best results on short, simple sentences

## Future Improvements

- Add attention mechanism
- Implement beam search decoding
- Explore transformer architecture
- Add BLEU score evaluation

## License

MIT License

## Acknowledgments

- Training data from [Tatoeba Project](https://tatoeba.org/)
- Pre-trained embeddings from [TensorFlow Hub](https://tfhub.dev/)
- Based on TensorFlow Specialization course on Coursera
