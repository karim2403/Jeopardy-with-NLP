"""Generate with an attention GRU and optionally inspect token-level attention."""
import argparse
from pathlib import Path
import sys

import torch

from ...tokenization import JeopardyTokenizer
from ..gru_seq2seq.checkpoint import load_checkpoint, sha256
from ..gru_seq2seq.dataset import format_source
from ..gru_seq2seq.model import select_device
from ..gru_seq2seq.precision import autocast_context, resolve_precision
from .model import GRUAttentionSeq2Seq, VectorizedGRUAttentionSeq2Seq


class Predictor:
    def __init__(self, checkpoint, device='auto'):
        checkpoint = Path(checkpoint)
        payload = load_checkpoint(checkpoint)
        self.config = payload['config']
        model_type = self.config.get('model_type')
        classes = {'gru_attention': GRUAttentionSeq2Seq,
                   'gru_attention_vectorized': VectorizedGRUAttentionSeq2Seq}
        if model_type not in classes:
            raise ValueError('Checkpoint is not a supported attention model')
        self.device = select_device(device)
        directory = checkpoint.parent/'tokenizer'
        if sha256(directory/'tokenizer.json') != self.config['tokenizer_sha256']:
            raise ValueError('Tokenizer does not match checkpoint')
        self.tokenizer = JeopardyTokenizer.load(directory)
        self.model = classes[model_type](**self.config['model']).to(self.device)
        self.model.load_state_dict(payload['model'])
        self.model.eval()
        self.precision = resolve_precision(self.device, self.config.get('precision', 'off'))
        self.pieces = {i: token for token, i in self.tokenizer.vocabulary().items()}

    def predict(self, category, text, task='answer', max_new_tokens=None, attention_top_k=3):
        if task not in ('answer', 'clue') or self.config['task'] not in (task, 'joint'):
            raise ValueError('Checkpoint was not trained for this task')
        if not category.strip() or not text.strip() or attention_top_k < 1:
            raise ValueError('Category/text must be nonempty and attention_top_k positive')
        source_text = format_source(task, category, text)
        ids = self.tokenizer.encode(source_text, add_bos=True, add_eos=True,
                                    max_length=self.config['max_source_length'], truncation=False)
        source = torch.tensor([ids], dtype=torch.long, device=self.device)
        limit = self.config['max_target_length'] - 1 if max_new_tokens is None else max_new_tokens
        with autocast_context(self.device, self.precision):
            output, attention = self.model.generate(
                source, torch.tensor([len(ids)]), self.tokenizer.bos_token_id,
                self.tokenizer.eos_token_id, self.tokenizer.pad_token_id,
                self.tokenizer.unk_token_id, limit, return_attention=True)
        output_ids = output[0].cpu().tolist()
        weights = attention[0].cpu()
        source_pieces = [self.pieces[i] for i in ids]
        trace = []
        for target_id, row in zip(output_ids, weights):
            top = torch.topk(row, min(attention_top_k, len(ids)))
            trace.append({
                'generated_id': target_id, 'generated_piece': self.pieces[target_id],
                'source_attention': [
                    {'position': position, 'piece': source_pieces[position], 'weight': weight}
                    for weight, position in zip(top.values.tolist(), top.indices.tolist())
                ],
            })
        return {'text': self.tokenizer.decode(output_ids), 'token_ids': output_ids,
                'reached_eos': self.tokenizer.eos_token_id in output_ids,
                'source_text': source_text, 'source_pieces': source_pieces,
                'attention': trace}


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--category', required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--clue')
    group.add_argument('--answer')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--max-new-tokens', type=int)
    parser.add_argument('--show-attention', action='store_true')
    parser.add_argument('--attention-top-k', type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(4)
    predictor = Predictor(args.checkpoint, args.device)
    task = 'answer' if args.clue is not None else 'clue'
    result = predictor.predict(args.category, args.clue if task == 'answer' else args.answer,
                               task, args.max_new_tokens, args.attention_top_k)
    print(f"{'Answer' if task == 'answer' else 'Clue'}: {result['text']}")
    if args.show_attention:
        print('Attention by generated BPE token (top source positions):')
        for step in result['attention']:
            sources = ', '.join(f"{item['piece']!r}={item['weight']:.3f}"
                                for item in step['source_attention'])
            print(f"  {step['generated_piece']!r}: {sources}")
    if not result['reached_eos']:
        print('Generation reached its token limit without EOS.')


if __name__ == '__main__':
    main()
