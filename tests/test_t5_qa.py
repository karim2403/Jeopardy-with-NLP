import unittest

from src.jeopardy.models.t5_qa.common import exact_match
from src.jeopardy.models.t5_qa.dataset import T5QADataset, format_input


class TestT5QA(unittest.TestCase):
    def test_input_format(self):
        self.assertEqual(format_input('HISTORY', 'A clue'),
                         'answer jeopardy clue\ncategory: HISTORY\nclue: A clue')

    def test_dataset_preserves_unrestricted_answer(self):
        rows = [{'id': '1', 'category': 'SCIENCE', 'clue': 'A clue',
                 'answer': 'An Answer Never Seen Before'}]
        item = T5QADataset(rows)[0]
        self.assertEqual(item['target_text'], 'An Answer Never Seen Before')
        self.assertEqual(len(T5QADataset(rows)), 1)

    def test_scoring_normalization(self):
        self.assertTrue(exact_match('Lincoln', 'Abraham Lincoln') is False)
        self.assertTrue(exact_match('The Netherlands', 'Netherlands'))


if __name__ == '__main__':
    unittest.main()
