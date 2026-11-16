import unittest

from src.jeopardy.models.gru_seq2seq.evaluate import exact_match, split_summary, token_f1


class TestEvaluationMetrics(unittest.TestCase):
    def test_exact_match_normalization(self):
        self.assertTrue(exact_match('The Paris!', 'Paris'))
        self.assertFalse(exact_match('Paris, Texas', 'Paris'))
        self.assertFalse(exact_match('', '!!!'))

    def test_token_f1(self):
        self.assertEqual(token_f1('New York', 'New York'), 1)
        self.assertAlmostEqual(token_f1('New', 'New York'), 2/3)
        self.assertEqual(token_f1('York', 'Paris'), 0)

    def test_empty_summary(self):
        self.assertEqual(split_summary([])['count'], 0)
        self.assertIsNone(split_summary([])['normalized_exact_match'])


if __name__ == '__main__':
    unittest.main()
