"""Evaluate one frozen checkpoint once on a held-out split; never trains."""
import argparse
from collections import Counter
from functools import partial
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ...evaluation import normalize_answer
from ...tokenization import JeopardyTokenizer
from .checkpoint import load_checkpoint, sha256
from .dataset import QADataset, LengthSortedBatchSampler, collate, load_rows
from .model import GRUSeq2Seq, select_device
from .progress import ProgressLine
from .precision import autocast_context, resolve_precision

ROOT = Path(__file__).resolve().parents[4]


def token_f1(prediction, target):
    """Bag-of-normalized-word F1; aliases and semantic equivalents still score zero."""
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(target).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if not overlap:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction, target):
    predicted, expected = normalize_answer(prediction), normalize_answer(target)
    return predicted == expected if expected else prediction.strip() == target.strip()


def split_summary(rows):
    count = len(rows)
    return {
        'count': count,
        'normalized_exact_match': sum(r['exact_match'] for r in rows) / count if count else None,
        'raw_exact_match': sum(r['raw_exact_match'] for r in rows) / count if count else None,
        'mean_token_f1': sum(r['token_f1'] for r in rows) / count if count else None,
        'nonempty_prediction_rate': sum(bool(r['prediction'].strip()) for r in rows) / count if count else None,
        'eos_rate': sum(r['reached_eos'] for r in rows) / count if count else None,
    }


def load_training_answers(path):
    answers = set()
    groups = set()
    with Path(path).open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            row = json.loads(line)
            if row.get('split') != 'train':
                raise ValueError(f'{path}:{number}: expected train split')
            answers.add(normalize_answer(row['answer']))
            groups.add(row['duplicate_group'])
    return answers, groups


@torch.inference_mode()
def evaluate(checkpoint, test_path, output_dir, device='auto', batch_size=128,
             max_new_tokens=None, progress_mode='auto', model_class=GRUSeq2Seq,
             expected_model_type='gru_seq2seq'):
    checkpoint, test_path, output_dir = Path(checkpoint), Path(test_path), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError('Output directory is not empty; test results are immutable. Choose a new directory.')
    if batch_size < 1 or (max_new_tokens is not None and max_new_tokens < 1):
        raise ValueError('Batch size and generation limit must be positive')
    payload = load_checkpoint(checkpoint)
    config = payload['config']
    model_type = config.get('model_type', 'gru_seq2seq')
    if model_type != expected_model_type:
        raise ValueError(f'Checkpoint model type is {model_type}, expected {expected_model_type}')
    if config['task'] not in ('answer', 'joint'):
        raise ValueError('Checkpoint was not trained for answer generation')
    tokenizer_dir = checkpoint.parent/'tokenizer'
    if sha256(tokenizer_dir/'tokenizer.json') != config['tokenizer_sha256']:
        raise ValueError('Tokenizer does not match checkpoint')
    tokenizer = JeopardyTokenizer.load(tokenizer_dir)
    selected_device = select_device(device)
    model = model_class(**config['model']).to(selected_device)
    model.load_state_dict(payload['model'])
    model.eval()

    test_rows = load_rows(test_path, 'test')
    training_answers, training_groups = load_training_answers(config['train'])
    overlap = training_groups & {row['duplicate_group'] for row in test_rows}
    if overlap:
        raise ValueError(f'Test contains {len(overlap)} clue groups also present in training')
    dataset = QADataset(test_rows, tokenizer, task='answer',
                        max_source_length=config['max_source_length'],
                        max_target_length=config['max_target_length'], truncate=False)
    loader = DataLoader(dataset, batch_sampler=LengthSortedBatchSampler(dataset, batch_size),
                        num_workers=0, collate_fn=partial(collate, pad_id=tokenizer.pad_token_id))
    generation_limit = max_new_tokens or config['max_target_length'] - 1
    resolved_precision = resolve_precision(selected_device, config.get('precision', 'off'))
    progress = ProgressLine(progress_mode)
    predictions = []
    loss_sum = target_tokens = 0
    started = time.perf_counter()
    for step, batch in enumerate(loader, 1):
        source = batch['source'].to(selected_device)
        target = batch['target'].to(selected_device)
        labels = target[:, 1:]
        with autocast_context(selected_device, resolved_precision):
            logits = model(source, batch['lengths'], target[:, :-1])
            batch_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                                         ignore_index=tokenizer.pad_token_id, reduction='sum')
            generated = model.generate(source, batch['lengths'], tokenizer.bos_token_id,
                                       tokenizer.eos_token_id, tokenizer.pad_token_id,
                                       tokenizer.unk_token_id, generation_limit)
        loss_sum += batch_loss.item()
        target_tokens += labels.ne(tokenizer.pad_token_id).sum().item()
        generated = generated.cpu().tolist()
        for item, output in zip(batch['items'], generated):
            prediction = tokenizer.decode(output)
            target_text = item['target_text']
            target_norm = normalize_answer(target_text)
            predictions.append({
                'id': item['id'], 'category': item['category'],
                'clue': item['clue'], 'expected': target_text,
                'prediction': prediction, 'prediction_token_ids': output,
                'exact_match': exact_match(prediction, target_text),
                'raw_exact_match': prediction.strip() == target_text.strip(),
                'token_f1': token_f1(prediction, target_text),
                'answer_seen_in_training': target_norm in training_answers,
                'reached_eos': tokenizer.eos_token_id in output,
            })
        elapsed = time.perf_counter() - started
        progress.update(f'Test {step}/{len(loader)} | {len(predictions)}/{len(dataset)} examples | '
                        f'{len(predictions)/elapsed:.0f} ex/s | ETA {elapsed/step*(len(loader)-step):.0f}s')
    progress.finish()
    teacher_loss = loss_sum / target_tokens
    seen = [row for row in predictions if row['answer_seen_in_training']]
    unseen = [row for row in predictions if not row['answer_seen_in_training']]
    summary = {
        'checkpoint': str(checkpoint.resolve()), 'checkpoint_sha256': sha256(checkpoint),
        'checkpoint_epoch': payload['epoch'], 'test_file': str(test_path.resolve()),
        'test_sha256': sha256(test_path), 'device': str(selected_device),
        'precision': resolved_precision,
        'batch_size': batch_size, 'max_new_tokens': generation_limit,
        'dataset': dataset.stats, 'teacher_forced_loss': teacher_loss,
        'teacher_forced_perplexity': math.exp(teacher_loss),
        'overall': split_summary(predictions),
        'answer_seen_in_training': split_summary(seen),
        'answer_unseen_in_training': split_summary(unseen),
        'elapsed_seconds': time.perf_counter() - started,
        'metric_notes': {
            'primary': 'normalized_exact_match',
            'normalization': 'casefold, punctuation removal, whitespace collapse, leading article removal',
            'token_f1': 'bag-of-normalized-whitespace-token F1; not semantic equivalence',
            'teacher_forced_loss': 'gold previous target tokens supplied; differs from free generation',
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir/'predictions.jsonl').open('x', encoding='utf-8') as stream:
        for row in predictions:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    (output_dir/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return summary


def print_summary(summary):
    overall, seen, unseen = (summary[key] for key in ('overall', 'answer_seen_in_training', 'answer_unseen_in_training'))
    print(f"Checkpoint epoch: {summary['checkpoint_epoch']} | test examples: {overall['count']:,} | device: {summary['device']}")
    print(f"Teacher-forced loss: {summary['teacher_forced_loss']:.4f} | perplexity: {summary['teacher_forced_perplexity']:.2f}")
    print(f"Overall normalized exact-match accuracy: {overall['normalized_exact_match']:.2%} "
          f"({round(overall['normalized_exact_match']*overall['count']):,}/{overall['count']:,})")
    print(f"Raw exact match: {overall['raw_exact_match']:.2%} | mean token F1: {overall['mean_token_f1']:.2%} | "
          f"EOS: {overall['eos_rate']:.2%} | nonempty: {overall['nonempty_prediction_rate']:.2%}")
    print(f"Seen answers: {seen['count']:,} ({seen['count']/overall['count']:.2%}) | exact match: {seen['normalized_exact_match']:.2%}")
    print(f"Unseen answers: {unseen['count']:,} ({unseen['count']/overall['count']:.2%}) | exact match: {unseen['normalized_exact_match']:.2%}")


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
    output = args.output_dir or args.checkpoint.parent/f'test_{args.checkpoint.stem}_epoch_{load_checkpoint(args.checkpoint)["epoch"]:03d}'
    summary = evaluate(args.checkpoint, args.test, output, args.device, args.batch_size,
                       args.max_new_tokens, args.progress)
    print_summary(summary)
    print(f'Results: {output}')


if __name__ == '__main__':
    main()
