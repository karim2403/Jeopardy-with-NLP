"""Read-only cosine-similarity inspection of saved GRU token embeddings."""
import argparse
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

from ...tokenization import JeopardyTokenizer
from .checkpoint import load_checkpoint, sha256


def cosine_neighbors(weights, vector, excluded=(), top_k=10):
    if top_k < 1:
        raise ValueError('top_k must be positive')
    if not torch.isfinite(vector).all() or vector.norm() <= 1e-12:
        raise ValueError('Cannot compare a zero or non-finite vector')
    valid = torch.isfinite(weights).all(dim=1) & (weights.norm(dim=1) > 1e-12)
    valid[list(excluded)] = False
    ids = valid.nonzero().flatten()
    scores = F.cosine_similarity(weights[ids], vector.unsqueeze(0), dim=1).clamp(-1, 1)
    # Stable ordering breaks score ties by vocabulary ID.
    order = torch.argsort(scores, descending=True, stable=True)[:top_k]
    return [(ids[i].item(), scores[i].item()) for i in order]


def pairwise_cosine(vectors):
    if not torch.isfinite(vectors).all() or (vectors.norm(dim=1) <= 1e-12).any():
        raise ValueError('Cannot compare zero or non-finite vectors')
    normalized = F.normalize(vectors, dim=1)
    return (normalized @ normalized.T).clamp(-1, 1).tolist()


class EmbeddingInspector:
    def __init__(self, checkpoint):
        checkpoint = Path(checkpoint)
        payload = load_checkpoint(checkpoint)
        directory = checkpoint.parent/'tokenizer'
        if sha256(directory/'tokenizer.json') != payload['config']['tokenizer_sha256']:
            raise ValueError('Tokenizer does not match checkpoint')
        self.tokenizer = JeopardyTokenizer.load(directory)
        self.weights = payload['model']['embedding.weight'].detach().float().cpu()
        if tuple(self.weights.shape) != (self.tokenizer.vocab_size, payload['config']['model']['embedding_dim']):
            raise ValueError('Embedding dimensions do not match tokenizer/configuration')
        self.pieces = {i: p for p, i in self.tokenizer.vocabulary().items()}
        self.special = {self.tokenizer.pad_token_id, self.tokenizer.unk_token_id,
                        self.tokenizer.bos_token_id, self.tokenizer.eos_token_id}
        self.checkpoint = str(checkpoint.resolve())
        self.epoch = payload['epoch']

    def represent(self, text, spacing='leading'):
        if spacing not in ('leading', 'none'):
            raise ValueError('spacing must be leading or none')
        text = text.strip()
        if not text:
            raise ValueError('Query text must be nonempty')
        encoded_text = (' ' if spacing == 'leading' else '') + text
        ids = self.tokenizer.encode(encoded_text)
        vector = self.weights[ids].mean(dim=0)
        info = dict(text=text, encoded_text=encoded_text, token_ids=ids,
                    pieces=[self.pieces[i] for i in ids],
                    representation='single token' if len(ids) == 1 else 'mean of subword embeddings')
        return info, vector

    def inspect(self, queries, compare=(), top_k=10, spacing='leading'):
        report = dict(checkpoint=self.checkpoint, epoch=self.epoch,
                      vocabulary_size=self.weights.size(0), embedding_dimension=self.weights.size(1),
                      spacing=spacing, queries=[], comparison=None)
        for text in queries:
            info, vector = self.represent(text, spacing)
            neighbors = cosine_neighbors(self.weights, vector, self.special | set(info['token_ids']), top_k)
            info['neighbors'] = [dict(token_id=i, piece=self.pieces[i], cosine=score) for i, score in neighbors]
            report['queries'].append(info)
        if compare:
            represented = [self.represent(text, spacing) for text in compare]
            report['comparison'] = dict(words=[info for info, _ in represented],
                                        cosine=pairwise_cosine(torch.stack([v for _, v in represented])))
        return report


def print_report(report):
    print(f"Checkpoint: {report['checkpoint']}\nEpoch: {report['epoch']} | Embeddings: "
          f"{report['vocabulary_size']:,} x {report['embedding_dimension']} | CPU inspection")
    print(f"Spacing: {report['spacing']} | cosine range: -1 to 1 (not a probability)")
    print('Neighbors are BPE tokens, not necessarily words. Special/query tokens are excluded.')
    for query in report['queries']:
        print(f"\nQuery: {query['text']!r}\nEncoded text: {query['encoded_text']!r}\n"
              f"Pieces: {query['pieces']}\nIDs: {query['token_ids']}\nRepresentation: {query['representation']}")
        print(f"{'Rank':>4}  {'Token piece':<28} {'ID':>6}  {'Cosine':>8}")
        for rank, neighbor in enumerate(query['neighbors'], 1):
            print(f"{rank:>4}  {neighbor['piece']!r:<28} {neighbor['token_id']:>6}  {neighbor['cosine']:>8.4f}")
    comparison = report['comparison']
    if comparison:
        print('\nPairwise cosine similarity (averaged subwords when needed):')
        for i, word in enumerate(comparison['words'], 1):
            print(f"  {i}: {word['text']!r} -> {word['pieces']}")
        print('        ' + ''.join(f'{i:>9}' for i in range(1, len(comparison['words'])+1)))
        for i, row in enumerate(comparison['cosine'], 1):
            print(f'{i:>7} ' + ''.join(f'{score:>9.3f}' for score in row))
    print('\nStatic task-trained embeddings; similarity does not establish meaning or answer accuracy.')
    print('Rare/unused vocabulary entries may be largely untrained. Ġ commonly marks a leading space.')


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--query', nargs='+', required=True, help='Words/phrases; quote multiword queries')
    parser.add_argument('--compare', nargs='+', default=[], help='Words/phrases for a pairwise matrix')
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--spacing', choices=['leading', 'none'], default='leading')
    parser.add_argument('--json', type=Path, help='Optional JSON destination (must not exist)')
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error('--top-k must be positive')
    if args.json and args.json.exists():
        parser.error('--json destination already exists; choose a new file')
    torch.set_num_threads(2)
    report = EmbeddingInspector(args.checkpoint).inspect(args.query, args.compare, args.top_k, args.spacing)
    print_report(report)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        print(f'JSON saved: {args.json}')


if __name__ == '__main__':
    main()
