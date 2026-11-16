"""Generate from a saved run without training or loading the dataset."""
import argparse
from pathlib import Path
import sys

import torch

from ...tokenization import JeopardyTokenizer
from .checkpoint import load_checkpoint, sha256
from .dataset import format_source
from .model import GRUSeq2Seq, select_device


class Predictor:
    def __init__(self, checkpoint, device='auto'):
        checkpoint = Path(checkpoint)
        payload = load_checkpoint(checkpoint)
        self.config = payload['config']
        self.device = select_device(device)
        directory = checkpoint.parent/'tokenizer'
        if sha256(directory/'tokenizer.json') != self.config['tokenizer_sha256']:
            raise ValueError('Tokenizer does not match checkpoint')
        self.tokenizer = JeopardyTokenizer.load(directory)
        self.model = GRUSeq2Seq(**self.config['model']).to(self.device)
        self.model.load_state_dict(payload['model'])
        self.model.eval()

    def predict(self, category, text, task='answer', max_new_tokens=None):
        if task not in ('answer', 'clue') or self.config['task'] not in (task, 'joint'):
            raise ValueError('Checkpoint was not trained for this task')
        if not category.strip() or not text.strip():
            raise ValueError('Category and input text must be nonempty')
        tokenizer = self.tokenizer
        ids = tokenizer.encode(format_source(task, category, text), add_bos=True, add_eos=True,
                               max_length=self.config['max_source_length'], truncation=False)
        source = torch.tensor([ids], dtype=torch.long, device=self.device)
        limit = self.config['max_target_length'] - 1 if max_new_tokens is None else max_new_tokens
        output = self.model.generate(source, torch.tensor([len(ids)]), tokenizer.bos_token_id,
                                     tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.unk_token_id, limit)
        output = output[0].cpu().tolist()
        return {'text': tokenizer.decode(output), 'token_ids': output,
                'reached_eos': tokenizer.eos_token_id in output}


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--category', required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--clue', help='Clue to answer')
    group.add_argument('--answer', help='Answer to generate a clue for')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--max-new-tokens', type=int)
    args = parser.parse_args()
    torch.set_num_threads(4)
    predictor = Predictor(args.checkpoint, args.device)
    task = 'answer' if args.clue is not None else 'clue'
    result = predictor.predict(args.category, args.clue if task == 'answer' else args.answer, task, args.max_new_tokens)
    print(f"Loaded {args.checkpoint} on {predictor.device}")
    print(f"{'Answer' if task == 'answer' else 'Clue'}: {result['text']}")
    if not result['reached_eos']:
        print('Generation reached its token limit without EOS.')


if __name__ == '__main__':
    main()
