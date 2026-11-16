import json
from pathlib import Path
import tempfile
import unittest

from src.jeopardy.tokenization import JeopardyTokenizer
from src.jeopardy.tokenization.compare import compare, load_records
from src.jeopardy.tokenization.metrics import evaluate, length_summary


class ComparisonTests(unittest.TestCase):
    def test_known_length_statistics(self):
        stats = length_summary([1, 2, 3, 100], 10)
        self.assertEqual(stats["p50"], 2)
        self.assertEqual(stats["p95"], 100)
        self.assertEqual(stats["tokens_per_whitespace_word"], 10.6)
        self.assertEqual(stats["over_limit_percent"]["32"], 25)
        self.assertEqual(stats["over_limit_percent"]["128"], 0)
        with self.assertRaises(ValueError):
            length_summary([], 0)

    def test_field_counts_and_special_token_accounting(self):
        tokenizer = JeopardyTokenizer(260).train(["ABC"])
        row = {"id": "one", "category": "A", "clue": "B", "answer": "C"}
        metrics = evaluate(tokenizer, [row])
        self.assertEqual(metrics["fields"]["clue"]["mean"], 1)
        self.assertEqual(metrics["fields"]["qa_input"]["mean"], len(tokenizer.encode("Category: A\nClue: B")) + 2)
        self.assertEqual(metrics["strings_checked"], 4)
        self.assertEqual(metrics["roundtrip_failures"], 0)

    def test_full_report_and_split_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for split in ("train", "validation"):
                path = root / f"{split}.jsonl"
                rows = [{"id": f"{split}-{i}", "duplicate_group": f"{split}-{i}", "split": split,
                         "category": "HISTORY", "clue": "A clue <script>alert(1)</script>", "answer": f"Name {i}"} for i in range(3)]
                path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
                paths[split] = path
            report = compare(paths['train'], paths['validation'], root / "report", vocab_sizes=[260, 280])
            self.assertEqual(len(report["candidates"]), 2)
            for candidate in report["candidates"]:
                self.assertEqual(candidate["validation"]["roundtrip_failures"], 0)
            html = (root / "report/report.html").read_text(encoding="utf-8")
            self.assertNotIn("<script>", html)
            self.assertIn("&lt;script&gt;", html)
            self.assertEqual(json.loads((root / "report/metrics.json").read_text(encoding="utf-8")), report)
            with self.assertRaises(ValueError):
                load_records(paths['validation'], "train")
            rows = load_records(paths['validation'], "validation")
            rows[0]['duplicate_group'] = 'train-0'
            paths['validation'].write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlap"):
                compare(paths['train'], paths['validation'], root / "bad", vocab_sizes=[260])


if __name__ == "__main__":
    unittest.main()
