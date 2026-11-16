import unittest

import torch

from src.jeopardy.models.gru_seq2seq.dataset import (LengthBucketBatchSampler,
                                                      LengthSortedBatchSampler)
from src.jeopardy.models.gru_seq2seq.precision import resolve_precision


class FakeDataset:
    def __init__(self, lengths):
        self.items = [{'source': torch.zeros(n, dtype=torch.long)} for n in lengths]

    def __len__(self):
        return len(self.items)


class OptimizationTests(unittest.TestCase):
    def test_bucket_sampler_covers_each_item_reproducibly(self):
        dataset = FakeDataset([30, 1, 29, 2, 28, 3, 27, 4, 26, 5])
        first = list(LengthBucketBatchSampler(
            dataset, 2, torch.Generator().manual_seed(9), 2))
        second = list(LengthBucketBatchSampler(
            dataset, 2, torch.Generator().manual_seed(9), 2))
        self.assertEqual(first, second)
        self.assertEqual(sorted(i for batch in first for i in batch), list(range(10)))
        self.assertTrue(all(len(batch) <= 2 for batch in first))

    def test_sorted_validation_batches(self):
        dataset = FakeDataset([2, 9, 1, 8, 3])
        batches = list(LengthSortedBatchSampler(dataset, 2))
        self.assertEqual(batches, [[1, 3], [4, 0], [2]])

    def test_precision_policy(self):
        self.assertEqual(resolve_precision(torch.device('cpu'), 'auto'), 'off')
        self.assertEqual(resolve_precision(torch.device('cpu'), 'off'), 'off')
        with self.assertRaises(RuntimeError):
            resolve_precision(torch.device('cpu'), 'fp16')
        expected = 'bf16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'fp16'
        if torch.cuda.is_available():
            self.assertEqual(resolve_precision(torch.device('cuda'), 'auto'), expected)


if __name__ == '__main__':
    unittest.main()
