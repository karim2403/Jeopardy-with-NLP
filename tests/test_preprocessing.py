import json
from pathlib import Path
import tempfile
import unittest

from src.jeopardy.evaluation import normalize_answer
from src.jeopardy.prepare_data import prepare
from src.jeopardy.preprocessing import clean_text, prepare_record
from src.jeopardy.splitting import assign_splits


def raw(clue="A clue", answer="A rare answer"):
    return {"category": "HISTORY", "question": clue, "answer": answer}


class PreprocessingTests(unittest.TestCase):
    def test_cleaning_preserves_model_text(self):
        self.assertEqual(clean_text("'  Mc\\'Donald's &amp; <i>Co.</i>\n  '") , "Mc'Donald's & Co.")
        self.assertEqual(clean_text("one<br>two"), "one two")
        self.assertEqual(clean_text("3 < 5"), "3 < 5")

    def test_media_removed_including_encoded_markup(self):
        for clue in ('<a href="x">here</a>', '&lt;img src="x"&gt;', 'see https://example.com', '<VIDEO>clip</VIDEO>'):
            self.assertEqual(prepare_record(raw(clue), 0)[1], "media_or_link")

    def test_missing_and_empty_text(self):
        self.assertEqual(prepare_record(raw(answer=None), 0)[1], "missing_or_invalid_required_text")
        self.assertEqual(prepare_record(raw(answer="<i> </i>"), 0)[1], "empty_cleaned_text")

    def test_scoring_never_changes_or_filters_answer(self):
        record, reason = prepare_record(raw(answer="!!!"), 0)
        self.assertIsNone(reason)
        self.assertEqual(record["answer"], "!!!")
        self.assertEqual(normalize_answer(record["answer"]), "")
        self.assertEqual(normalize_answer("The Ant"), "ant")
        self.assertEqual(normalize_answer("The Ant", remove_article=False), "the ant")

    def test_duplicates_grouped_and_conflicts_preserved(self):
        records = [prepare_record(raw(f"Clue {i}"), i)[0] for i in range(100)]
        records.append(prepare_record(raw("CLUE 0", "Different answer"), 100)[0])
        assign_splits(records)
        self.assertEqual(records[0]["split"], records[-1]["split"])
        self.assertTrue(records[0]["conflicting_answers"])
        self.assertEqual(len(records), 101)
        before = {r["id"]: r["split"] for r in records}
        assign_splits(list(reversed(records)))
        self.assertEqual(before, {r["id"]: r["split"] for r in records})

    def test_end_to_end_reproducibility_and_rare_answers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = [raw(f"Clue {i}", f"Unique answer {i}") for i in range(100)]
            source += [raw('<img src="x">'), raw(answer=None)]
            path = root / "input.json"
            path.write_text(json.dumps(source), encoding="utf-8")
            first = prepare(path, root / "one")
            second = prepare(path, root / "two")
            self.assertEqual(first, second)
            self.assertEqual(first["retained_rows"], 100)
            self.assertEqual(sum(first["removed_by_reason"].values()), 2)
            groups = []
            answers = set()
            for split in ("train", "validation", "test"):
                data = [json.loads(line) for line in (root / "one" / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
                groups.append({r["duplicate_group"] for r in data})
                answers.update(r["answer"] for r in data)
                self.assertTrue(all(r["split"] == split for r in data))
            self.assertEqual(len(answers), 100)
            self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])


if __name__ == "__main__":
    unittest.main()
