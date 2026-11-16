"""Evaluate a frozen attention-GRU checkpoint on the held-out test split."""
import argparse
from pathlib import Path
import sys

import torch

from ..gru_seq2seq.checkpoint import load_checkpoint
from ..gru_seq2seq.evaluate import ROOT, evaluate, print_summary
from .model import GRUAttentionSeq2Seq, VectorizedGRUAttentionSeq2Seq


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--test', type=Path, default=ROOT/'data/processed/test.jsonl')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--max-new-tokens', type=int)
    parser.add_argument('--progress', choices=['auto', 'line', 'plain'], default='auto')
    args = parser.parse_args()
    torch.set_num_threads(4)
    epoch = load_checkpoint(args.checkpoint)['epoch']
    model_type = load_checkpoint(args.checkpoint)['config'].get('model_type')
    classes = {'gru_attention': GRUAttentionSeq2Seq,
               'gru_attention_vectorized': VectorizedGRUAttentionSeq2Seq}
    if model_type not in classes:
        parser.error('checkpoint is not a supported attention model')
    output = args.output_dir or args.checkpoint.parent/f'test_{args.checkpoint.stem}_epoch_{epoch:03d}'
    summary = evaluate(args.checkpoint, args.test, output, args.device, args.batch_size,
                       args.max_new_tokens, args.progress, classes[model_type], model_type)
    print_summary(summary)
    print(f'Results: {output}')


if __name__ == '__main__':
    main()
