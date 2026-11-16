import unittest

import torch

from src.jeopardy.models.gru_attention.model import (GRUAttentionSeq2Seq,
                                                      VectorizedGRUAttentionSeq2Seq)


class GRUAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.source = torch.tensor([[2, 5, 6, 3, 0], [2, 7, 3, 0, 0]])
        self.lengths = torch.tensor([4, 3])
        self.target = torch.tensor([[2, 8, 9], [2, 10, 3]])

    def test_shapes_masking_and_gradient(self):
        model = GRUAttentionSeq2Seq(32, embedding_dim=8, hidden_dim=12,
                                    attention_dim=7, dropout=0)
        logits, attention = model(self.source, self.lengths, self.target, return_attention=True)
        self.assertEqual(logits.shape, (2, 3, 32))
        self.assertEqual(attention.shape, (2, 3, 5))
        torch.testing.assert_close(attention.sum(-1), torch.ones(2, 3))
        self.assertTrue((attention[0, :, 4] == 0).all())
        self.assertTrue((attention[1, :, 3:] == 0).all())
        logits.sum().backward()
        self.assertIsNotNone(model.attention.score.weight.grad)
        self.assertGreater(model.attention.score.weight.grad.norm().item(), 0)

    def test_extra_right_padding_does_not_change_eval_output(self):
        model = GRUAttentionSeq2Seq(32, embedding_dim=8, hidden_dim=12,
                                    attention_dim=7, dropout=.2).eval()
        expected = model(self.source, self.lengths, self.target)
        padded = torch.nn.functional.pad(self.source, (0, 4))
        actual = model(padded, self.lengths, self.target)
        torch.testing.assert_close(expected, actual)

    def test_bidirectional_multilayer_and_generation(self):
        model = GRUAttentionSeq2Seq(32, embedding_dim=8, hidden_dim=12,
                                    attention_dim=7, dropout=.1, bidirectional=True,
                                    encoder_layers=2, decoder_layers=2).eval()
        logits = model(self.source, self.lengths, self.target)
        self.assertEqual(logits.shape, (2, 3, 32))
        ids, attention = model.generate(self.source, self.lengths, 2, 3, 0, 1, 5,
                                        return_attention=True)
        self.assertEqual(ids.shape[0], 2)
        self.assertEqual(attention.shape, (2, ids.shape[1], 5))
        self.assertLessEqual(ids.shape[1], 5)

    def test_invalid_hyperparameters(self):
        with self.assertRaises(ValueError):
            GRUAttentionSeq2Seq(32, attention_dim=0)
        with self.assertRaises(ValueError):
            GRUAttentionSeq2Seq(32, dropout=1)

    def test_vectorized_shapes_masking_gradient_and_padding(self):
        model = VectorizedGRUAttentionSeq2Seq(
            32, embedding_dim=8, hidden_dim=12, attention_dim=7, dropout=0).eval()
        logits, attention = model(self.source, self.lengths, self.target,
                                  return_attention=True)
        self.assertEqual(logits.shape, (2, 3, 32))
        self.assertEqual(attention.shape, (2, 3, 5))
        torch.testing.assert_close(attention.sum(-1), torch.ones(2, 3))
        self.assertTrue((attention[1, :, 3:] == 0).all())
        expected = logits.detach()
        actual = model(torch.nn.functional.pad(self.source, (0, 3)),
                       self.lengths, self.target)
        torch.testing.assert_close(expected, actual)
        model.train()
        model(self.source, self.lengths, self.target).sum().backward()
        self.assertGreater(model.attention.keys.weight.grad.norm().item(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA not available')
    def test_vectorized_bf16_cuda_forward_backward(self):
        model = VectorizedGRUAttentionSeq2Seq(
            32, embedding_dim=8, hidden_dim=12, attention_dim=7).cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = model(self.source.cuda(), self.lengths, self.target.cuda()).float().sum()
        loss.backward()
        self.assertTrue(torch.isfinite(model.output.weight.grad).all())


if __name__ == '__main__':
    unittest.main()
