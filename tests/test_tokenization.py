import json
from pathlib import Path
import tempfile
import unittest

from src.jeopardy.tokenization import JeopardyTokenizer
from src.jeopardy.train_tokenizer import training_texts


class TokenizerTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = JeopardyTokenizer(vocab_size=300).train([
            "HISTORY", "This president lived here.", "Abraham Lincoln",
            "This city is in France.", "Paris", "This president lived here.",
        ])

    def test_unseen_unicode_round_trip(self):
        for text in ["", "Paris", "paris", "  McDonald's & Co.\n", "Zürich 東京 🦉", "a\tb\n"]:
            ids = self.tokenizer.encode(text)
            self.assertNotIn(self.tokenizer.unk_token_id, ids)
            self.assertEqual(self.tokenizer.decode(ids), text)
        self.assertNotEqual(self.tokenizer.encode("Paris"), self.tokenizer.encode("paris"))

    def test_padding_and_special_tokens(self):
        batch = self.tokenizer.batch_encode(["Paris", "A longer sentence"], add_bos=True, add_eos=True)
        self.assertEqual(len(batch["input_ids"][0]), len(batch["input_ids"][1]))
        for text, ids, mask in zip(["Paris", "A longer sentence"], batch["input_ids"], batch["attention_mask"]):
            self.assertEqual(ids[0], self.tokenizer.bos_token_id)
            self.assertEqual(ids[sum(mask)-1], self.tokenizer.eos_token_id)
            self.assertTrue(all(i == self.tokenizer.pad_token_id for i, m in zip(ids, mask) if m == 0))
            self.assertEqual(self.tokenizer.decode(ids), text)
        fixed = self.tokenizer.batch_encode(["Paris"], padding="max_length", max_length=30)
        self.assertEqual(len(fixed["input_ids"][0]), 30)
        ragged = self.tokenizer.batch_encode(["", "Paris"], padding=False)
        self.assertEqual(ragged["input_ids"][0], [])
        self.assertEqual(self.tokenizer.batch_encode([]), {"input_ids": [], "attention_mask": []})

    def test_explicit_truncation_preserves_eos(self):
        with self.assertRaises(ValueError):
            self.tokenizer.encode("A long sentence", max_length=3)
        ids = self.tokenizer.encode("A long sentence", max_length=3, truncation=True, add_bos=True, add_eos=True)
        self.assertEqual(len(ids), 3)
        self.assertEqual(ids[-1], self.tokenizer.eos_token_id)
        # A capped call must not change later calls.
        self.assertGreater(len(self.tokenizer.encode("A long sentence")), 3)
        with self.assertRaises(ValueError):
            self.tokenizer.encode("x", truncation=True)
        with self.assertRaises(ValueError):
            self.tokenizer.encode("x", max_length=1, add_bos=True, add_eos=True)

    def test_save_load_and_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            self.tokenizer.save(directory)
            loaded = JeopardyTokenizer.load(directory)
            self.assertEqual(loaded.encode("Zürich 🦉"), self.tokenizer.encode("Zürich 🦉"))
            self.assertEqual(loaded.training_metadata, self.tokenizer.training_metadata)
            self.assertEqual(loaded.vocab_size, self.tokenizer.vocab_size)
            path = Path(directory) / "tokenizer.json"
            path.write_bytes(path.read_bytes() + b" ")
            with self.assertRaises(ValueError):
                JeopardyTokenizer.load(directory)

    def test_invalid_lifecycle_and_configuration(self):
        with self.assertRaises(RuntimeError):
            JeopardyTokenizer().encode("x")
        with self.assertRaises(RuntimeError):
            self.tokenizer.train(["new vocabulary"])
        for size in [0, 259, True, 300.5]:
            with self.assertRaises(ValueError):
                JeopardyTokenizer(vocab_size=size)
        with self.assertRaises(ValueError):
            JeopardyTokenizer().train([])
        with self.assertRaises(ValueError):
            self.tokenizer.encode("literal <eos>")
        with self.assertRaises(ValueError):
            self.tokenizer.decode([self.tokenizer.vocab_size])

    def test_training_loader_rejects_held_out_data(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            row = {"split": "train", "category": "HISTORY", "clue": "A clue", "answer": "Rare name"}
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            self.assertEqual(list(training_texts(path)), ["HISTORY", "A clue", "Rare name"])
            for split in ["test", "validation", None]:
                row["split"] = split
                path.write_text(json.dumps(row) + "\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    list(training_texts(path))


if __name__ == "__main__":
    unittest.main()
