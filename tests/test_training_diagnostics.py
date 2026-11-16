import io
import unittest
from unittest.mock import patch

import torch

from src.jeopardy.models.gru_seq2seq.embedding_probes import EmbeddingProbes
from src.jeopardy.models.gru_seq2seq.progress import ProgressLine


class StubTokenizer:
    def encode(self, text):
        return {' man': [0], ' woman': [1], 'two pieces': [0, 1]}[text]

    def token_pieces(self, text):
        return [str(i) for i in self.encode(text)]


class DiagnosticsTests(unittest.TestCase):
    def test_probe_reference_and_resume_without_rng_changes(self):
        weights = torch.tensor([[1., 0.], [0., 1.]])
        before = torch.get_rng_state().clone()
        probes = EmbeddingProbes(StubTokenizer(), [['man', 'woman']])
        initial = probes.measure(weights, 0)
        self.assertEqual(initial['pairs'][0]['cosine'], 0)
        self.assertIsNone(initial['pairs'][0]['delta_previous'])
        # Change the fixture directly, not through training.
        updated = torch.tensor([[1., 0.], [1., 0.]])
        measured = probes.measure(updated, 1)
        self.assertEqual(measured['pairs'][0]['delta_previous'], 1)
        resumed = EmbeddingProbes(StubTokenizer(), [['man', 'woman']], probes.state)
        result = resumed.measure(weights, 2)
        self.assertEqual(result['reference_epoch'], 0)
        self.assertEqual(result['pairs'][0]['delta_previous'], -1)
        self.assertEqual(result['pairs'][0]['delta_reference'], 0)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_old_checkpoint_gets_resume_reference(self):
        probes = EmbeddingProbes(StubTokenizer(), [['man', 'woman']])
        result = probes.measure(torch.eye(2), 10)
        self.assertEqual(result['reference_epoch'], 10)

    def test_zero_vector_is_undefined(self):
        probes = EmbeddingProbes(StubTokenizer(), [['man', 'woman']])
        self.assertIsNone(probes.measure(torch.zeros(2, 2), 0)['pairs'][0]['cosine'])

    def test_inline_overwrites_and_clears_old_tail(self):
        stream = io.StringIO()
        with patch('shutil.get_terminal_size', return_value=__import__('os').terminal_size((80, 24))):
            progress = ProgressLine('line', stream)
            progress.update('longer line')
            progress.update('short')
            progress.finish()
        self.assertEqual(stream.getvalue(), '\rlonger line\rshort      \n')

    def test_redirected_output_stays_plain(self):
        stream = io.StringIO()
        progress = ProgressLine('auto', stream)
        progress.update('first')
        progress.update('second')
        progress.finish()
        self.assertEqual(stream.getvalue(), 'first\nsecond\n')


if __name__ == '__main__':
    unittest.main()
