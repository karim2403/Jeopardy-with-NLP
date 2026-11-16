"""Evaluate a frozen T5 checkpoint on validation or a one-time test split."""
import argparse
from collections import Counter
from functools import partial
import json
import math
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from ...evaluation import normalize_answer
from ..gru_seq2seq.checkpoint import sha256
from ..gru_seq2seq.dataset import load_rows
from ..gru_seq2seq.evaluate import split_summary, token_f1
from ..gru_seq2seq.progress import ProgressLine
from .common import autocast_context, exact_match, load_run_config, model_device, resolve_precision
from .dataset import T5Collator, T5QADataset, to_device

ROOT = Path(__file__).resolve().parents[4]


def load_training_answers(path):
    answers = set()
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            answers.add(normalize_answer(json.loads(line)['answer']))
    return answers


@torch.inference_mode()
def evaluate(checkpoint, data_path, split, output_dir, device='auto', precision='auto',
             batch_size=32, max_new_tokens=32, num_beams=1, progress_mode='auto'):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    checkpoint, data_path, output_dir = map(Path, (checkpoint, data_path, output_dir))
    if split not in ('validation', 'test'):
        raise ValueError('split must be validation or test')
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError('Output directory is not empty; choose a new directory')
    config = load_run_config(checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    selected_device = model_device(device)
    resolved_precision = resolve_precision(selected_device, precision)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(selected_device).eval()
    rows = load_rows(data_path, split)
    dataset = T5QADataset(rows)
    collator = T5Collator(tokenizer, config['max_input_length'], config['max_target_length'])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator, num_workers=0)
    training_answers = load_training_answers(config['train'])
    predictions = []
    loss_sum = examples = 0
    progress = ProgressLine(progress_mode)
    started = time.perf_counter()
    for step, batch in enumerate(loader, 1):
        items = batch.pop('items')
        tensors = to_device(batch, selected_device)
        with autocast_context(selected_device, resolved_precision):
            loss = model(**tensors).loss
            generation_inputs = {key: tensors[key] for key in ('input_ids', 'attention_mask')}
            generated = model.generate(**generation_inputs, max_new_tokens=max_new_tokens,
                                       num_beams=num_beams, do_sample=False)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        loss_sum += loss.item() * len(items)
        examples += len(items)
        for item, prediction in zip(items, decoded):
            expected = item['target_text']
            predictions.append({
                'id': item['id'], 'category': item['category'], 'clue': item['clue'],
                'expected': expected, 'prediction': prediction,
                'exact_match': exact_match(prediction, expected),
                'raw_exact_match': prediction.strip() == expected.strip(),
                'token_f1': token_f1(prediction, expected),
                'answer_seen_in_training': normalize_answer(expected) in training_answers,
                'reached_eos': True,
            })
        elapsed = time.perf_counter() - started
        progress.update(f'{split} {step}/{len(loader)} | {examples}/{len(dataset)} | '
                        f'{examples/elapsed:.0f} ex/s | ETA {elapsed/step*(len(loader)-step):.0f}s')
    progress.finish()
    seen = [row for row in predictions if row['answer_seen_in_training']]
    unseen = [row for row in predictions if not row['answer_seen_in_training']]
    summary = {
        'checkpoint': str(checkpoint.resolve()), 'data_file': str(data_path.resolve()),
        'data_sha256': sha256(data_path), 'split': split, 'device': str(selected_device),
        'precision': resolved_precision, 'batch_size': batch_size, 'num_beams': num_beams,
        'max_new_tokens': max_new_tokens, 'teacher_forced_loss': loss_sum/examples,
        'teacher_forced_perplexity': math.exp(loss_sum/examples),
        'overall': split_summary(predictions),
        'answer_seen_in_training': split_summary(seen),
        'answer_unseen_in_training': split_summary(unseen),
        'unique_normalized_predictions': len({normalize_answer(r['prediction']) for r in predictions}),
        'elapsed_seconds': time.perf_counter() - started,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    with (output_dir/'predictions.jsonl').open('x', encoding='utf-8') as stream:
        for row in predictions:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--split', choices=['validation', 'test'], default='validation')
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--precision', choices=['auto', 'off', 'fp16', 'bf16'], default='auto')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--max-new-tokens', type=int, default=32)
    parser.add_argument('--num-beams', type=int, default=1)
    parser.add_argument('--progress', choices=['auto', 'line', 'plain'], default='auto')
    args = parser.parse_args()
    if min(args.batch_size, args.max_new_tokens, args.num_beams) < 1:
        parser.error('batch size, generation length, and beam count must be positive')
    data = args.data or ROOT/f'data/processed/{args.split}.jsonl'
    output = args.output_dir or args.checkpoint.parent/f'{args.split}_evaluation_beam_{args.num_beams}'
    summary = evaluate(args.checkpoint, data, args.split, output, args.device, args.precision,
                       args.batch_size, args.max_new_tokens, args.num_beams, args.progress)
    overall = summary['overall']
    print(f"{args.split}: {overall['count']:,} | exact {overall['normalized_exact_match']:.2%} | "
          f"token F1 {overall['mean_token_f1']:.2%} | loss {summary['teacher_forced_loss']:.4f} | "
          f"unique predictions {summary['unique_normalized_predictions']:,}")
    print(f'Results: {output}')


if __name__ == '__main__':
    main()
