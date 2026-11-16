import unittest

from src.jeopardy.models.validation_ablation import paired_comparison, shuffled_field


def prediction(record_id, text, correct=False):
    return {'id': record_id, 'prediction': text, 'exact_match': correct}


class TestValidationAblation(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {'id': str(i), 'category': f'category {i}', 'clue': f'clue {i}', 'answer': f'answer {i}'}
            for i in range(10)
        ]

    def test_shuffle_is_deterministic_and_preserves_other_fields(self):
        first, matches = shuffled_field(self.rows, 'category', 17)
        second, _ = shuffled_field(self.rows, 'category', 17)
        self.assertEqual(first, second)
        self.assertEqual(matches, 0)
        self.assertCountEqual([row['category'] for row in first],
                              [row['category'] for row in self.rows])
        self.assertEqual([row['clue'] for row in first], [row['clue'] for row in self.rows])
        self.assertEqual([row['answer'] for row in first], [row['answer'] for row in self.rows])

    def test_paired_comparison_matches_by_id_not_order(self):
        intact = [prediction('a', 'right', True), prediction('b', 'same', False)]
        ablated = [prediction('b', 'same', True), prediction('a', 'wrong', False)]
        result = paired_comparison(intact, ablated)
        self.assertEqual(result['normalized_prediction_change_rate'], .5)
        self.assertEqual(result['correct_to_incorrect'], 1)
        self.assertEqual(result['incorrect_to_correct'], 1)
        self.assertEqual(result['net_exact_match_change'], 0)

    def test_paired_comparison_rejects_different_examples(self):
        with self.assertRaises(ValueError):
            paired_comparison([prediction('a', 'x')], [prediction('b', 'x')])


if __name__ == '__main__':
    unittest.main()
