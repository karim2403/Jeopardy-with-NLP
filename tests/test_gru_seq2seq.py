from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch.nn import functional as F

from src.jeopardy.tokenization import JeopardyTokenizer
from src.jeopardy.models.gru_seq2seq.checkpoint import load_checkpoint
from src.jeopardy.models.gru_seq2seq.dataset import QADataset, collate
from src.jeopardy.models.gru_seq2seq.model import GRUSeq2Seq, select_device
from src.jeopardy.models.gru_seq2seq.predict import Predictor
from src.jeopardy.models.gru_seq2seq.train import DEFAULTS, run


def row(index, split='train'):
    return dict(id=f'{split}-{index}', duplicate_group=f'{split}-{index}', split=split,
                category='CAPITALS', clue=('France' if index % 2 == 0 else 'Italy'),
                answer=('Paris' if index % 2 == 0 else 'Rome'))


class GRUTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.tokenizer = JeopardyTokenizer(280).train(['CAPITALS France Paris Italy Rome'] * 4)

    def test_padding_does_not_change_encoder_state(self):
        torch.manual_seed(0)
        model = GRUSeq2Seq(self.tokenizer.vocab_size, embedding_dim=8, hidden_dim=12).eval()
        source = torch.tensor([[2, 10, 3]])
        expected = model.encode(source, torch.tensor([3]))
        actual = model.encode(F.pad(source, (0, 5)), torch.tensor([3]))
        torch.testing.assert_close(expected, actual)

    def test_task_formatting_and_caps(self):
        dataset = QADataset([row(0)], self.tokenizer, task='joint')
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0]['target_text'], 'Paris')
        self.assertEqual(dataset[1]['target_text'], 'France')
        with self.assertRaises(ValueError):
            QADataset([row(0)], self.tokenizer, max_source_length=2)
        short = QADataset([row(0)], self.tokenizer, max_source_length=2, truncate=True)
        self.assertEqual(short[0]['source'].tolist(), [2, 3])

    def test_tiny_batch_can_be_learned(self):
        torch.manual_seed(7)
        t = self.tokenizer
        dataset = QADataset([row(0), row(1)], t)
        batch = collate(dataset.items, t.pad_token_id)
        model = GRUSeq2Seq(t.vocab_size, embedding_dim=24, hidden_dim=48)
        optimizer = torch.optim.Adam(model.parameters(), lr=.02)
        losses = []
        for step in range(180):
            logits = model(batch['source'], batch['lengths'], batch['target'][:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, t.vocab_size), batch['target'][:, 1:].reshape(-1), ignore_index=0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        model.eval()
        ids = model.generate(batch['source'], batch['lengths'], 2, 3, 0, 1, 20)
        self.assertLess(losses[-1], losses[0] * .05)
        self.assertEqual([t.decode(x) for x in ids.tolist()], ['Paris', 'Rome'])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA not available')
    def test_gpu_forward_backward_and_cpu_equivalence(self):
        torch.manual_seed(0)
        cpu = GRUSeq2Seq(self.tokenizer.vocab_size, embedding_dim=8, hidden_dim=12)
        gpu = GRUSeq2Seq(self.tokenizer.vocab_size, embedding_dim=8, hidden_dim=12).cuda()
        gpu.load_state_dict(cpu.state_dict())
        batch = collate(QADataset([row(0), row(1)], self.tokenizer).items, 0)
        expected = cpu(batch['source'], batch['lengths'], batch['target'][:, :-1])
        # Compare full FP32 arithmetic, not the GPU's optional TF32 approximation.
        previous = torch.backends.cudnn.allow_tf32
        torch.backends.cudnn.allow_tf32 = False
        try:
            actual = gpu(batch['source'].cuda(), batch['lengths'], batch['target'][:, :-1].cuda())
            torch.testing.assert_close(expected, actual.cpu(), atol=1e-5, rtol=1e-4)
            actual.sum().backward()
        finally:
            torch.backends.cudnn.allow_tf32 = previous
        self.assertIsNotNone(gpu.output.weight.grad)

    def test_checkpoint_prediction_and_exact_cpu_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.tokenizer.save(root/'tokenizer')
            for split in ('train', 'validation'):
                (root/f'{split}.jsonl').write_text('\n'.join(json.dumps(row(i, split)) for i in range(4)), encoding='utf-8')

            def args(name, epochs, resume=None):
                config = dict.fromkeys(DEFAULTS)
                config.update(train=None, validation=None, tokenizer=None, run_dir=None,
                              resume=resume, epochs=epochs, device='cpu', threads=2, log_every=1)
                if resume is None:
                    config.update(train=root/'train.jsonl', validation=root/'validation.jsonl', tokenizer=root/'tokenizer',
                                  run_dir=root/name, embedding_dim=8, hidden_dim=12, batch_size=2, eval_samples=2)
                return SimpleNamespace(**config)

            with redirect_stdout(io.StringIO()):
                run(args('full', 2))
                run(args('resumed', 1))
                run(args('resumed', 2, root/'resumed/last.pt'))
            full = load_checkpoint(root/'full/last.pt')
            resumed = load_checkpoint(root/'resumed/last.pt')
            for key in full['model']:
                torch.testing.assert_close(full['model'][key], resumed['model'][key], rtol=0, atol=0)
            predictor = Predictor(root/'resumed/best.pt', device='cpu')
            output = predictor.predict('CAPITALS', 'France')
            self.assertIsInstance(output['text'], str)
            with self.assertRaises(ValueError):
                predictor.predict('CAPITALS', 'Paris', task='clue')
            with self.assertRaises(ValueError), redirect_stdout(io.StringIO()):
                run(args('full', 2))

    def test_device_override(self):
        self.assertEqual(select_device('cpu').type, 'cpu')
        self.assertEqual(select_device('auto').type, 'cuda' if torch.cuda.is_available() else 'cpu')


if __name__ == '__main__':
    unittest.main()
