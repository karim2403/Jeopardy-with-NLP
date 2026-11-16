import unittest
import torch

from src.jeopardy.models.gru_seq2seq.inspect_embeddings import cosine_neighbors, pairwise_cosine


class EmbeddingInspectionTests(unittest.TestCase):
    def test_neighbors_exclude_self_special_and_zero(self):
        vectors = torch.tensor([[0., 0.], [1., 0.], [2., 0.], [0., 1.], [-1., 0.]])
        neighbors = cosine_neighbors(vectors, vectors[1], excluded={1}, top_k=20)
        self.assertEqual([i for i, _ in neighbors], [2, 3, 4])
        for (_, actual), expected in zip(neighbors, [1, 0, -1]):
            self.assertAlmostEqual(actual, expected)

    def test_matrix_symmetry_and_direction(self):
        matrix = pairwise_cosine(torch.tensor([[1., 0.], [0., 2.], [-1., 0.]]))
        self.assertEqual(matrix, [[1., 0., -1.], [0., 1., 0.], [-1., 0., 1.]])

    def test_zero_vector_rejected(self):
        with self.assertRaises(ValueError):
            pairwise_cosine(torch.zeros(1, 2))
        with self.assertRaises(ValueError):
            cosine_neighbors(torch.ones(2, 2), torch.zeros(2))


if __name__ == '__main__':
    unittest.main()
