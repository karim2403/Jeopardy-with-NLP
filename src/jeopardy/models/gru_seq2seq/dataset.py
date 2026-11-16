"""Task formatting, split validation, tokenization, and batch collation."""
import json
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data import Sampler


def format_source(task, category, text):
    if task not in ("answer", "clue"):
        raise ValueError("Task must be answer or clue")
    field = "clue" if task == "answer" else "answer"
    return f"{task}: category: {category}\n{field}: {text}"


def load_rows(path, expected_split):
    rows = []
    with open(path, encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            row = json.loads(line)
            if row.get("split") != expected_split:
                raise ValueError(f"{path}:{number}: expected {expected_split} split")
            for field in ("id", "duplicate_group", "category", "clue", "answer"):
                if not isinstance(row.get(field), str) or not row[field]:
                    raise ValueError(f"{path}:{number}: missing {field}")
            rows.append(row)
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError("Dataset is empty or contains duplicate record IDs")
    return rows


def check_disjoint(train, validation):
    for field in ("id", "duplicate_group"):
        if {r[field] for r in train} & {r[field] for r in validation}:
            raise ValueError(f"Train/validation overlap in {field}")


def sample_rows(rows, limit, seed):
    if limit is None:
        return rows
    return random.Random(seed).sample(rows, min(limit, len(rows)))


class QADataset(Dataset):
    def __init__(self, rows, tokenizer, task="answer", max_source_length=128,
                 max_target_length=64, truncate=False):
        if task not in ("answer", "clue", "joint"):
            raise ValueError("Invalid task")
        if min(max_source_length, max_target_length) < 2:
            raise ValueError("Length caps must fit BOS/EOS")
        self.items = []
        self.stats = {"records": len(rows), "examples": 0, "source_over_cap": 0, "target_over_cap": 0}
        for row in rows:
            for direction in (("answer", "clue") if task == "joint" else (task,)):
                text = row["clue"] if direction == "answer" else row["answer"]
                target_text = row["answer"] if direction == "answer" else row["clue"]
                source_text = format_source(direction, row['category'], text)
                source = tokenizer.encode(source_text, add_bos=True, add_eos=True)
                target = tokenizer.encode(target_text, add_bos=True, add_eos=True)
                self.stats["source_over_cap"] += len(source) > max_source_length
                self.stats["target_over_cap"] += len(target) > max_target_length
                if truncate:
                    source = source[:max_source_length-1] + [tokenizer.eos_token_id] if len(source) > max_source_length else source
                    target = target[:max_target_length-1] + [tokenizer.eos_token_id] if len(target) > max_target_length else target
                self.items.append({"source": torch.tensor(source), "target": torch.tensor(target),
                                   "id": row['id'], "task": direction,
                                   "category": row['category'], "clue": row['clue'],
                                   "source_text": source_text, "target_text": target_text})
        self.stats["examples"] = len(self.items)
        if not truncate and (self.stats['source_over_cap'] or self.stats['target_over_cap']):
            raise ValueError(f"Length caps exceeded: {self.stats}. Increase caps or explicitly use --truncate.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate(items, pad_id):
    return {
        "source": pad_sequence([x['source'] for x in items], batch_first=True, padding_value=pad_id),
        "lengths": torch.tensor([len(x['source']) for x in items]),
        "target": pad_sequence([x['target'] for x in items], batch_first=True, padding_value=pad_id),
        "items": items,
    }


class LengthBucketBatchSampler(Sampler):
    """Seeded shuffled batches with similar source lengths to reduce padding."""
    def __init__(self, dataset, batch_size, generator, bucket_multiplier=50):
        if min(batch_size, bucket_multiplier) < 1:
            raise ValueError('Batch size and bucket multiplier must be positive')
        self.lengths = [len(item['source']) for item in dataset.items]
        self.batch_size = batch_size
        self.generator = generator
        self.pool_size = batch_size * bucket_multiplier

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        order = torch.randperm(len(self.lengths), generator=self.generator).tolist()
        batches = []
        for start in range(0, len(order), self.pool_size):
            pool = sorted(order[start:start+self.pool_size],
                          key=lambda index: self.lengths[index], reverse=True)
            batches.extend(pool[i:i+self.batch_size]
                           for i in range(0, len(pool), self.batch_size))
        shuffled = torch.randperm(len(batches), generator=self.generator).tolist()
        yield from (batches[index] for index in shuffled)


class LengthSortedBatchSampler(Sampler):
    """Deterministic validation batches with minimal source padding."""
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size
        self.order = sorted(range(len(dataset)),
                            key=lambda index: len(dataset.items[index]['source']), reverse=True)

    def __len__(self):
        return (len(self.order) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        yield from (self.order[i:i+self.batch_size]
                    for i in range(0, len(self.order), self.batch_size))
