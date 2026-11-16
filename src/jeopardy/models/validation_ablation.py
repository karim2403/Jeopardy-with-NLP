"""Measure how much a frozen answer model depends on clue and category text.

This is deliberately validation-only.  Each condition keeps the original answer
target while replacing one input field with a value from another validation row.
"""
import argparse
from functools import partial
import json
import math
from pathlib import Path
import random
import sys
import time

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ..evaluation import normalize_answer
from ..tokenization import JeopardyTokenizer
from .gru_attention.model import GRUAttentionSeq2Seq, VectorizedGRUAttentionSeq2Seq
from .gru_seq2seq.checkpoint import load_checkpoint, sha256
from .gru_seq2seq.dataset import QADataset, LengthSortedBatchSampler, collate, load_rows
from .gru_seq2seq.evaluate import exact_match, split_summary, token_f1
from .gru_seq2seq.model import GRUSeq2Seq, select_device
from .gru_seq2seq.precision import autocast_context, resolve_precision
from .gru_seq2seq.progress import ProgressLine

ROOT = Path(__file__).resolve().parents[3]
MODEL_CLASSES = {
    'gru_seq2seq': GRUSeq2Seq,
    'gru_attention': GRUAttentionSeq2Seq,
    'gru_attention_vectorized': VectorizedGRUAttentionSeq2Seq,
}


def shuffled_field(rows, field, seed):
    """Return copied rows whose field values come from a low-match permutation."""
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    best = indices[:]
    best_matches = len(rows) + 1
    for _ in range(100):
        candidate = indices[:]
        rng.shuffle(candidate)
        matches = sum(rows[i][field] == rows[j][field] for i, j in enumerate(candidate))
        if matches < best_matches:
            best, best_matches = candidate, matches
        if matches == 0:
            break
    changed = []
    for index, donor in enumerate(best):
        row = dict(rows[index])
        row[field] = rows[donor][field]
        changed.append(row)
    return changed, best_matches


def condition_summary(predictions, teacher_loss):
    summary = split_summary(predictions)
    summary['teacher_forced_loss'] = teacher_loss
    summary['teacher_forced_perplexity'] = math.exp(teacher_loss)
    summary['unique_normalized_predictions'] = len({normalize_answer(r['prediction']) for r in predictions})
    return summary


@torch.inference_mode()
def evaluate_condition(model, rows, tokenizer, config, device, precision, batch_size,
                       generation_limit, progress_mode, label):
    dataset = QADataset(rows, tokenizer, task='answer',
                        max_source_length=config['max_source_length'],
                        max_target_length=config['max_target_length'], truncate=False)
    loader = DataLoader(dataset, batch_sampler=LengthSortedBatchSampler(dataset, batch_size),
                        num_workers=0, collate_fn=partial(collate, pad_id=tokenizer.pad_token_id))
    progress = ProgressLine(progress_mode)
    predictions = []
    loss_sum = target_tokens = 0
    started = time.perf_counter()
    for step, batch in enumerate(loader, 1):
        source = batch['source'].to(device)
        target = batch['target'].to(device)
        labels = target[:, 1:]
        with autocast_context(device, precision):
            logits = model(source, batch['lengths'], target[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                                   ignore_index=tokenizer.pad_token_id, reduction='sum')
            generated = model.generate(source, batch['lengths'], tokenizer.bos_token_id,
                                       tokenizer.eos_token_id, tokenizer.pad_token_id,
                                       tokenizer.unk_token_id, generation_limit)
        loss_sum += loss.item()
        target_tokens += labels.ne(tokenizer.pad_token_id).sum().item()
        for item, output in zip(batch['items'], generated.cpu().tolist()):
            prediction = tokenizer.decode(output)
            expected = item['target_text']
            predictions.append({
                'id': item['id'], 'category': item['category'], 'clue': item['clue'],
                'expected': expected, 'prediction': prediction,
                'exact_match': exact_match(prediction, expected),
                'raw_exact_match': prediction.strip() == expected.strip(),
                'token_f1': token_f1(prediction, expected),
                'reached_eos': tokenizer.eos_token_id in output,
            })
        elapsed = time.perf_counter() - started
        progress.update(f'{label} {step}/{len(loader)} | {len(predictions)}/{len(dataset)} | '
                        f'{len(predictions)/elapsed:.0f} ex/s')
    progress.finish()
    return predictions, condition_summary(predictions, loss_sum / target_tokens)


def paired_comparison(intact, ablated):
    intact_by_id = {row['id']: row for row in intact}
    ablated_by_id = {row['id']: row for row in ablated}
    if intact_by_id.keys() != ablated_by_id.keys():
        raise ValueError('Ablation conditions do not contain the same record IDs')
    pairs = [(intact_by_id[key], ablated_by_id[key]) for key in intact_by_id]
    count = len(pairs)
    changed = sum(normalize_answer(a['prediction']) != normalize_answer(b['prediction']) for a, b in pairs)
    lost = sum(a['exact_match'] and not b['exact_match'] for a, b in pairs)
    gained = sum(not a['exact_match'] and b['exact_match'] for a, b in pairs)
    return {
        'count': count,
        'normalized_prediction_change_rate': changed / count,
        'correct_to_incorrect': lost,
        'incorrect_to_correct': gained,
        'net_exact_match_change': (gained - lost) / count,
    }


def run_ablation(checkpoint, validation, output_dir, device='auto', batch_size=128,
                 max_new_tokens=None, limit=None, seed=42, progress_mode='auto'):
    checkpoint, validation, output_dir = map(Path, (checkpoint, validation, output_dir))
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError('Output directory is not empty; choose a new directory')
    if batch_size < 1 or (limit is not None and limit < 1):
        raise ValueError('Batch size and validation limit must be positive')
    payload = load_checkpoint(checkpoint)
    config = payload['config']
    model_type = config.get('model_type', 'gru_seq2seq')
    if model_type not in MODEL_CLASSES:
        raise ValueError(f'Unsupported model type: {model_type}')
    if config['task'] not in ('answer', 'joint'):
        raise ValueError('Checkpoint was not trained for answer generation')
    tokenizer_dir = checkpoint.parent / 'tokenizer'
    if sha256(tokenizer_dir/'tokenizer.json') != config['tokenizer_sha256']:
        raise ValueError('Tokenizer does not match checkpoint')
    tokenizer = JeopardyTokenizer.load(tokenizer_dir)
    selected_device = select_device(device)
    model = MODEL_CLASSES[model_type](**config['model']).to(selected_device)
    model.load_state_dict(payload['model'])
    model.eval()
    precision = resolve_precision(selected_device, config.get('precision', 'off'))

    rows = load_rows(validation, 'validation')
    if limit is not None:
        rows = random.Random(seed).sample(rows, min(limit, len(rows)))
    category_rows, category_matches = shuffled_field(rows, 'category', seed + 1)
    clue_rows, clue_matches = shuffled_field(rows, 'clue', seed + 2)
    conditions = {'intact': rows, 'shuffled_category': category_rows, 'shuffled_clue': clue_rows}
    generation_limit = max_new_tokens or config['max_target_length'] - 1
    all_predictions, summaries = {}, {}
    started = time.perf_counter()
    for label, condition_rows in conditions.items():
        predictions, summary = evaluate_condition(
            model, condition_rows, tokenizer, config, selected_device, precision,
            batch_size, generation_limit, progress_mode, label)
        all_predictions[label], summaries[label] = predictions, summary
    comparisons = {label: paired_comparison(all_predictions['intact'], all_predictions[label])
                   for label in ('shuffled_category', 'shuffled_clue')}
    result = {
        'checkpoint': str(checkpoint.resolve()), 'checkpoint_sha256': sha256(checkpoint),
        'checkpoint_epoch': payload['epoch'], 'model_type': model_type,
        'validation_file': str(validation.resolve()), 'validation_sha256': sha256(validation),
        'sample_seed': seed, 'sample_limit': limit, 'sample_count': len(rows),
        'device': str(selected_device), 'precision': precision, 'batch_size': batch_size,
        'max_new_tokens': generation_limit,
        'shuffle_unchanged_values': {'category': category_matches, 'clue': clue_matches},
        'conditions': summaries, 'paired_vs_intact': comparisons,
        'elapsed_seconds': time.perf_counter() - started,
        'notes': {
            'target': 'The original answer is retained in every condition.',
            'interpretation': 'A larger degradation after shuffling indicates greater dependence on that field.',
            'scope': 'Validation diagnostics only; no weights are updated and the test set is not read.',
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir/'summary.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    for label, predictions in all_predictions.items():
        with (output_dir/f'{label}.jsonl').open('x', encoding='utf-8') as stream:
            for row in predictions:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    return result


def print_summary(result):
    print(f"Model: {result['model_type']} | epoch {result['checkpoint_epoch']} | "
          f"validation examples: {result['sample_count']:,} | device: {result['device']}")
    for label, summary in result['conditions'].items():
        print(f"{label:>19}: exact {summary['normalized_exact_match']:.2%} | "
              f"token F1 {summary['mean_token_f1']:.2%} | loss {summary['teacher_forced_loss']:.4f} | "
              f"unique predictions {summary['unique_normalized_predictions']:,}")
    for label, comparison in result['paired_vs_intact'].items():
        print(f"{label:>19} vs intact: prediction changed {comparison['normalized_prediction_change_rate']:.2%} | "
              f"correct→wrong {comparison['correct_to_incorrect']:,} | "
              f"wrong→correct {comparison['incorrect_to_correct']:,} | "
              f"net exact {comparison['net_exact_match_change']:+.2%}")


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--validation', type=Path, default=ROOT/'data/processed/validation.jsonl')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--max-new-tokens', type=int)
    parser.add_argument('--limit', type=int, help='Seeded validation subset size; default evaluates all rows')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--progress', choices=['auto', 'line', 'plain'], default='auto')
    args = parser.parse_args()
    torch.set_num_threads(4)
    epoch = load_checkpoint(args.checkpoint)['epoch']
    output = args.output_dir or args.checkpoint.parent/f'validation_ablation_epoch_{epoch:03d}'
    result = run_ablation(args.checkpoint, args.validation, output, args.device, args.batch_size,
                          args.max_new_tokens, args.limit, args.seed, args.progress)
    print_summary(result)
    print(f'Results: {output}')


if __name__ == '__main__':
    main()
