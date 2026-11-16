"""Run with python -m src.jeopardy.data.prepare from the repository root."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from ..evaluation import normalize_answer
from .preprocessing import PIPELINE_VERSION, prepare_record
from .splitting import assign_splits

ROOT = Path(__file__).resolve().parents[3]


def prepare(input_path, output_dir, seed=42):
    input_path, output_dir = Path(input_path), Path(output_dir)
    payload = input_path.read_bytes()
    source = json.loads(payload)
    if not isinstance(source, list):
        raise ValueError("Input must be a JSON array of records")
    records, removed = [], Counter()
    for index, raw in enumerate(source):
        record, reason = prepare_record(raw, index)
        if reason:
            removed[reason] += 1
        else:
            record["answer_normalized"] = normalize_answer(record["answer"])
            records.append(record)
    if not records:
        raise ValueError("No usable records remain after preprocessing")
    assign_splits(records, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for split in ("train", "validation", "test"):
        path = output_dir / f"{split}.jsonl"
        digest, count = hashlib.sha256(), 0
        with path.open("wb") as stream:
            for record in records:
                if record["split"] == split:
                    line = (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
                    stream.write(line)
                    digest.update(line)
                    count += 1
        outputs[split] = {"file": path.name, "rows": count, "sha256": digest.hexdigest()}
    group_counts = Counter(r["duplicate_group"] for r in records)
    manifest = {
        "pipeline_version": PIPELINE_VERSION, "source_file": input_path.name,
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "settings": {"seed": seed, "split_ratios": [0.8, 0.1, 0.1], "remove_media": True,
                     "minimum_answer_frequency": None, "tokenization": False,
                     "scoring_remove_leading_article": True},
        "source_rows": len(source), "retained_rows": len(records),
        "removed_by_reason": dict(sorted(removed.items())),
        "duplicate_groups": sum(count > 1 for count in group_counts.values()),
        "conflicting_answer_groups": len({r["duplicate_group"] for r in records if r["conflicting_answers"]}),
        "outputs": outputs,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "JEOPARDY_QUESTIONS1.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(prepare(args.input, args.output_dir, args.seed), indent=2))


if __name__ == "__main__":
    main()
