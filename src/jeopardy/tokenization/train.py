"""Train on processed training records only; no validation/test access."""
import argparse
import hashlib
import json
from pathlib import Path

from . import JeopardyTokenizer

ROOT = Path(__file__).resolve().parents[3]
FIELDS = ("category", "clue", "answer")


def training_texts(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            if row.get("split") != "train":
                raise ValueError(f"Row {line_number} is not a training record")
            for field in FIELDS:
                text = row.get(field)
                if not isinstance(text, str) or not text:
                    raise ValueError(f"Row {line_number}: missing or empty {field}")
                yield text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/processed/train.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/tokenizer")
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()
    tokenizer = JeopardyTokenizer(args.vocab_size, args.min_frequency)
    tokenizer.train(training_texts(args.input), show_progress=True)
    with args.input.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    tokenizer.training_metadata.update(source_file=args.input.name, source_sha256=digest, fields=list(FIELDS))
    tokenizer.save(args.output_dir)
    print(f"Saved {tokenizer.vocab_size:,} tokens to {args.output_dir}")


if __name__ == "__main__":
    main()
