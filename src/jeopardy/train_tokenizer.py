"""Compatibility entry point; prefer src.jeopardy.tokenization.train."""
from .tokenization.train import main, training_texts

if __name__ == "__main__":
    main()
