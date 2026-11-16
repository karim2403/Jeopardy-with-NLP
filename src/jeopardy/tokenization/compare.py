"""Compare train-only BPE vocabularies on validation; never read test data."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import time

import tokenizers

from . import JeopardyTokenizer
from .metrics import benchmark, evaluate, training_usage
from .report import write_report
from .train import FIELDS, ROOT, training_texts


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_records(path, split):
    with Path(path).open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if not rows:
        raise ValueError(f"Empty {split} dataset")
    for row in rows:
        if row.get("split") != split:
            raise ValueError(f"Expected only {split} records")
        for field in (*FIELDS, "id", "duplicate_group"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"Missing/non-string {field}")
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError(f"Duplicate IDs in {split}")
    return rows


def inspection_examples(rows):
    examples = [{"label": "Validation clue", "text": rows[0]["clue"]},
                {"label": "Validation answer", "text": rows[0]["answer"]}]
    examples.extend([
        {"label": "Synthetic names / Unicode probe", "text": "Zürich, São Paulo, 東京, and 🦉"},
        {"label": "Synthetic dates / abbreviations probe", "text": "In 1865, the U.S. president was Abraham Lincoln."},
        {"label": "Synthetic punctuation / wordplay probe", "text": '"Before & After": McDonald\'s, O\'Brien, and a 3-letter word.'},
    ])
    return examples


def compare(train_path, validation_path, output_dir, vocab_sizes=(4000, 8000, 16000), min_frequency=2):
    if not vocab_sizes or len(set(vocab_sizes)) != len(vocab_sizes):
        raise ValueError("Provide distinct vocabulary sizes")
    for size in vocab_sizes:
        JeopardyTokenizer(size, min_frequency)  # validate before creating artifacts
    train = load_records(train_path, "train")
    validation = load_records(validation_path, "validation")
    for key in ("id", "duplicate_group"):
        if {r[key] for r in train} & {r[key] for r in validation}:
            raise ValueError(f"Train/validation overlap in {key}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "tokenizers_version": tokenizers.__version__,
        "train": {"file": str(Path(train_path).resolve()), "sha256": file_hash(train_path), "rows": len(train)},
        "validation": {"file": str(Path(validation_path).resolve()), "sha256": file_hash(validation_path), "rows": len(validation)},
        "settings": {"vocab_sizes": list(vocab_sizes), "min_frequency": min_frequency,
                     "training_fields": list(FIELDS), "benchmark_seed": 42,
                     "embedding_dimension_example": 128, "percentile_method": "nearest-rank"},
        "candidates": [],
    }
    # Identical shuffled sample and order for all candidates; no test data read.
    indices = random.Random(42).sample(range(len(validation)), min(2000, len(validation)))
    speed_texts = [validation[i]["clue"] for i in indices]
    examples = inspection_examples(validation)
    del train  # reload lazily for training and usage counts
    for size in vocab_sizes:
        print(f"Training and evaluating {size:,} tokens...", flush=True)
        tokenizer = JeopardyTokenizer(size, min_frequency)
        start = time.perf_counter()
        tokenizer.train(training_texts(train_path))
        train_seconds = time.perf_counter() - start
        tokenizer.training_metadata.update(source_file=Path(train_path).name,
                                           source_sha256=report["train"]["sha256"], fields=list(FIELDS))
        directory = output_dir / f"vocab_{size}"
        tokenizer.save(directory)
        # Evaluate the serialized artifact users will actually load.
        tokenizer = JeopardyTokenizer.load(directory)
        candidate = {
            "requested_vocab_size": size, "actual_vocab_size": tokenizer.vocab_size,
            "training_seconds": train_seconds,
            "artifact_bytes": sum(p.stat().st_size for p in directory.glob("*.json")),
            "tokenizer_sha256": file_hash(directory / "tokenizer.json"),
            "embedding_parameters_128d": tokenizer.vocab_size * 128,
            "validation": evaluate(tokenizer, validation),
            "training_usage": training_usage(tokenizer, training_texts(train_path)),
            "benchmark": benchmark(tokenizer, speed_texts),
            "examples": [dict(e, pieces=tokenizer.token_pieces(e["text"]),
                              ids=tokenizer.encode(e["text"]),
                              decoded=tokenizer.decode(tokenizer.encode(e["text"]))) for e in examples],
        }
        report["candidates"].append(candidate)
    if file_hash(train_path) != report["train"]["sha256"] or file_hash(validation_path) != report["validation"]["sha256"]:
        raise RuntimeError("Dataset changed during comparison; rerun with stable input files")
    write_report(report, output_dir)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "data/processed/train.jsonl")
    parser.add_argument("--validation", type=Path, default=ROOT / "data/processed/validation.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/tokenizer_comparison")
    parser.add_argument("--vocab-sizes", nargs="+", type=int, default=[4000, 8000, 16000])
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()
    compare(args.train, args.validation, args.output_dir, args.vocab_sizes, args.min_frequency)
    print(f"Report: {args.output_dir / 'report.html'}")


if __name__ == "__main__":
    main()
