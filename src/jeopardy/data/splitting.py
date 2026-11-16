"""Seeded splitting with all copies of a clue kept together."""
from collections import defaultdict
import random


def assign_splits(records, seed=42):
    groups = defaultdict(list)
    for record in records:
        groups[record["duplicate_group"]].append(record)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    seen = 0
    for key in keys:
        group = groups[key]
        midpoint = seen + len(group) / 2
        split = "train" if midpoint < len(records) * .8 else (
            "validation" if midpoint < len(records) * .9 else "test"
        )
        conflict = len({r["answer"].casefold() for r in group}) > 1
        for record in group:
            record["split"] = split
            record["conflicting_answers"] = conflict
        seen += len(group)
    return records
