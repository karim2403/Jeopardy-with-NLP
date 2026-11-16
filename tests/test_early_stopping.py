import unittest

from src.jeopardy.models.gru_seq2seq.early_stopping import EarlyStopping


class EarlyStoppingTests(unittest.TestCase):
    def test_stops_after_patience(self):
        stopping = EarlyStopping(patience=2)
        self.assertTrue(stopping.update(3.0)['improved'])
        self.assertFalse(stopping.update(3.1)['should_stop'])
        self.assertTrue(stopping.update(3.2)['should_stop'])

    def test_improvement_resets_counter_and_min_delta(self):
        stopping = EarlyStopping(patience=2, min_delta=.1)
        stopping.update(3.0)
        self.assertFalse(stopping.update(2.95)['improved'])
        result = stopping.update(2.8)
        self.assertTrue(result['improved'])
        self.assertEqual(result['bad_epochs'], 0)

    def test_state_round_trip_and_disable(self):
        first = EarlyStopping(patience=3)
        first.update(2.0)
        first.update(2.1)
        restored = EarlyStopping(patience=3, state=first.state_dict())
        self.assertEqual(restored.bad_epochs, 1)
        disabled = EarlyStopping(patience=0)
        disabled.update(1.0)
        self.assertFalse(disabled.update(2.0)['should_stop'])


if __name__ == '__main__':
    unittest.main()
