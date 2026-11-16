"""Small fixed cosine probes; read-only, no RNG or model forward passes."""
import math

import torch
from torch.nn import functional as F

DEFAULT_PAIRS = [['man', 'woman'], ['Man', 'Woman'], ['Paris', 'London'], ['France', 'England']]


class EmbeddingProbes:
    def __init__(self, tokenizer, pairs, state=None):
        self.pairs = pairs
        self.words = {}
        for pair in pairs:
            if len(pair) != 2 or any(not word.strip() for word in pair):
                raise ValueError('Each probe requires two nonempty words/phrases')
            for word in pair:
                text = ' ' + word.strip()
                self.words[word] = {'ids': tokenizer.encode(text), 'pieces': tokenizer.token_pieces(text)}
        self.state = state if state and state['pairs'] == pairs else None

    @torch.inference_mode()
    def measure(self, weights, epoch):
        vectors = {word: weights[data['ids']].detach().float().mean(0) for word, data in self.words.items()}
        values = []
        for left, right in self.pairs:
            a, b = vectors[left], vectors[right]
            value = F.cosine_similarity(a, b, dim=0).clamp(-1, 1).item()
            values.append(value if math.isfinite(value) and a.norm() > 1e-12 and b.norm() > 1e-12 else None)
        old = self.state
        baseline = old['baseline'] if old else values
        delta = lambda a, b: a-b if a is not None and b is not None else None
        report = {'epoch': epoch, 'reference_epoch': old['reference_epoch'] if old else epoch,
                  'spacing': 'leading', 'words': self.words, 'pairs': [
                      dict(left=left, right=right, cosine=value,
                           delta_previous=delta(value, old['previous'][i]) if old else None,
                           delta_reference=delta(value, baseline[i]))
                      for i, ((left, right), value) in enumerate(zip(self.pairs, values))]}
        self.state = dict(pairs=self.pairs, baseline=baseline, previous=values,
                          reference_epoch=report['reference_epoch'], epoch=epoch)
        return report


def print_probes(report, show_pieces=False):
    reference = 'initialization' if report['reference_epoch'] == 0 else f"resume epoch {report['reference_epoch']}"
    print(f"Embedding cosine probes | epoch {report['epoch']} | reference: {reference} (not an accuracy metric)", flush=True)
    if show_pieces:
        for word, data in report['words'].items():
            print(f"  {word!r}: {data['pieces']} | IDs {data['ids']}")
        print('  Consistent leading space; multi-piece words use mean embeddings.')
    print(f"  {'Pair':<30} {'Cosine':>9} {'Change prev':>12} {'Change ref':>12}")
    for pair in report['pairs']:
        fmt = lambda value: '--' if value is None else f'{value:+.4f}'
        label = f"{pair['left']} / {pair['right']}"
        print(f"  {label:<30} {fmt(pair['cosine']):>9} {fmt(pair['delta_previous']):>12} {fmt(pair['delta_reference']):>12}", flush=True)
