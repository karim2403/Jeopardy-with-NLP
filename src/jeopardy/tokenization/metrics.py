"""Tokenizer diagnostics. These measure representation, not answer accuracy."""
from collections import Counter
import math
import statistics
import time


def length_summary(lengths, word_count, limits=(16, 32, 64, 96, 128)):
    if not lengths:
        raise ValueError("Cannot summarize an empty corpus")
    ordered = sorted(lengths)
    return {
        "count": len(lengths), "total_tokens": sum(lengths),
        "mean": statistics.mean(lengths),
        **{f"p{p}": ordered[math.ceil(p / 100 * len(ordered)) - 1] for p in (50, 90, 95, 99)},
        "max": ordered[-1],
        "tokens_per_whitespace_word": sum(lengths) / word_count if word_count else None,
        "over_limit_percent": {str(n): 100 * sum(v > n for v in lengths) / len(lengths) for n in limits},
    }


def evaluate(tokenizer, records):
    fields = ("category", "clue", "answer", "qa_input")
    lengths = {field: [] for field in fields}
    words = Counter()
    unknown, failures, total = 0, 0, 0
    longest = {}
    for row in records:
        texts = {f: row[f] for f in fields[:-1]}
        texts["qa_input"] = f"Category: {row['category']}\nClue: {row['clue']}"
        for field, text in texts.items():
            # QA input includes BOS/EOS; raw fields have no special tokens.
            ids = tokenizer.encode(text, add_bos=field == "qa_input", add_eos=field == "qa_input")
            lengths[field].append(len(ids))
            words[field] += len(text.split())
            total += len(ids)
            unknown += ids.count(tokenizer.unk_token_id)
            failures += tokenizer.decode(ids) != text
            if field not in longest or len(ids) > longest[field]["tokens"]:
                longest[field] = {"id": row["id"], "text": text, "tokens": len(ids)}
    return {
        "fields": {f: length_summary(lengths[f], words[f]) for f in fields},
        "strings_checked": len(records) * len(fields), "roundtrip_failures": failures,
        "unknown_tokens": unknown, "unknown_token_rate": unknown / total if total else 0,
        "longest": longest,
    }


def training_usage(tokenizer, texts):
    counts = Counter()
    for text in texts:
        counts.update(tokenizer.encode(text))
    special = {tokenizer.pad_token_id, tokenizer.unk_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id}
    values = [counts[i] for i in range(tokenizer.vocab_size) if i not in special]
    return {"ordinary_tokens": len(values), "unused_tokens": values.count(0),
            "used_fewer_than_10_times": sum(0 < v < 10 for v in values)}


def benchmark(tokenizer, texts, repeats=3):
    """Median single-string encode loop; excludes disk I/O and decoding."""
    if not texts or repeats < 1:
        raise ValueError("Benchmark requires texts and at least one repeat")
    for text in texts[:100]:
        tokenizer.encode(text)
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        for text in texts:
            tokenizer.encode(text)
        times.append(time.perf_counter() - start)
    median = statistics.median(times)
    return {"strings": len(texts), "repeats": repeats, "median_seconds": median,
            "strings_per_second": len(texts) / median,
            "utf8_mb_per_second": sum(len(t.encode('utf-8')) for t in texts) / median / 1_000_000}
