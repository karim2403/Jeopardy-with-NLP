"""Fine-tune a pretrained T5-family model on standardized Jeopardy data."""
import argparse
import json
import math
from pathlib import Path
import random
import sys
import time

import torch
from torch.utils.data import DataLoader, Subset

from ..gru_seq2seq.checkpoint import sha256
from ..gru_seq2seq.dataset import check_disjoint, load_rows, sample_rows
from ..gru_seq2seq.early_stopping import EarlyStopping
from ..gru_seq2seq.progress import ProgressLine
from .common import (autocast_context, exact_match, load_training_state, model_device,
                     resolve_precision, restore_rng, rng_state, save_checkpoint)
from .dataset import T5Collator, T5QADataset, to_device

ROOT = Path(__file__).resolve().parents[4]
DEFAULTS = dict(model_name='google/flan-t5-small', batch_size=16,
                gradient_accumulation_steps=4, learning_rate=5e-5, weight_decay=.01,
                warmup_ratio=.05, seed=42, max_input_length=128, max_target_length=32,
                train_limit=None, validation_limit=None, eval_samples=1000,
                early_stopping_patience=2, early_stopping_min_delta=0.0,
                precision='auto', num_beams=1)


def append_metric(path, entry):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(entry, ensure_ascii=False) + '\n')


def epoch_loss(model, loader, device, precision, epoch, optimizer=None, scheduler=None,
               scaler=None, accumulation_steps=1, log_every=50, progress_mode='auto'):
    training = optimizer is not None
    model.train(training)
    total_loss = total_examples = 0
    progress = ProgressLine(progress_mode)
    started = time.perf_counter()
    if training:
        optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training):
        for step, batch in enumerate(loader, 1):
            items = batch.pop('items')
            tensors = to_device(batch, device)
            with autocast_context(device, precision):
                loss = model(**tensors).loss
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss; the last completed checkpoint remains available')
            if training:
                scaled_loss = loss / accumulation_steps
                if scaler and scaler.is_enabled():
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                update = step % accumulation_steps == 0 or step == len(loader)
                if update:
                    if scaler and scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if scaler and scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
            count = len(items)
            total_loss += loss.item() * count
            total_examples += count
            if step == 1 or step % log_every == 0 or step == len(loader):
                elapsed = time.perf_counter() - started
                eta = elapsed / step * (len(loader) - step)
                stage = 'train' if training else 'validation'
                progress.update(f'E{epoch} {stage} {step}/{len(loader)} | loss {total_loss/total_examples:.4f} | '
                                f'{total_examples/elapsed:.0f} ex/s | ETA {eta:.0f}s')
    progress.finish()
    return total_loss / total_examples


@torch.inference_mode()
def generated_metrics(model, dataset, tokenizer, collator, indices, device, precision,
                      batch_size, max_target_length, num_beams):
    loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False,
                        collate_fn=collator, num_workers=0)
    model.eval()
    rows = []
    for batch in loader:
        items = batch.pop('items')
        batch.pop('labels')
        tensors = to_device(batch, device)
        with autocast_context(device, precision):
            generated = model.generate(**tensors, max_new_tokens=max_target_length,
                                       num_beams=num_beams, do_sample=False)
        predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for item, prediction in zip(items, predictions):
            rows.append({'id': item['id'], 'input': item['input_text'],
                         'expected': item['target_text'], 'predicted': prediction,
                         'exact_match': exact_match(prediction, item['target_text'])})
    return sum(row['exact_match'] for row in rows) / len(rows), rows


def run(args):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
    torch.set_num_threads(args.threads)
    device = model_device(args.device)
    resume_dir = Path(args.resume).resolve() if args.resume else None
    resumed = load_training_state(resume_dir) if resume_dir else None
    if resumed:
        if resume_dir.name != 'last':
            raise ValueError('Resume from the run\'s last directory')
        run_dir = resume_dir.parent
        config = json.loads((run_dir/'config.json').read_text(encoding='utf-8'))
        for key in (*DEFAULTS, 'train', 'validation'):
            supplied = getattr(args, key, None)
            if supplied is not None and key in config and str(supplied) != str(config[key]):
                raise ValueError(f'Cannot change {key} when resuming; start a new run')
    else:
        config = {key: getattr(args, key) if getattr(args, key) is not None else value
                  for key, value in DEFAULTS.items()}
        config.update(train=str((args.train or ROOT/'data/processed/train.jsonl').resolve()),
                      validation=str((args.validation or ROOT/'data/processed/validation.jsonl').resolve()),
                      run_dir=str((args.run_dir or ROOT/'artifacts/models/t5_qa/flan_t5_small_v1').resolve()),
                      model_type='t5_qa')
        run_dir = Path(config['run_dir'])
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ValueError('Run directory is not empty; choose a new --run-dir or --resume its last directory')
    epochs = args.epochs if args.epochs is not None else config.get('epochs', 5)
    if epochs <= (resumed['epoch'] if resumed else 0):
        raise ValueError('--epochs is total epochs and must exceed the saved epoch')
    config['epochs'] = epochs
    positive = ('batch_size', 'gradient_accumulation_steps', 'max_input_length',
                'max_target_length', 'eval_samples', 'num_beams')
    if any(config[key] < 1 for key in positive):
        raise ValueError(f'{positive} must be positive')
    if config['learning_rate'] <= 0 or config['weight_decay'] < 0 or not 0 <= config['warmup_ratio'] < 1:
        raise ValueError('Invalid optimizer settings')
    precision = resolve_precision(device, config['precision'])
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    print(f'Device: {device}' + (f' ({torch.cuda.get_device_name(device)})' if device.type == 'cuda' else '') +
          f' | precision: {precision}', flush=True)
    print(f"Loading {'checkpoint' if resumed else 'pretrained model'}: "
          f"{resume_dir if resumed else config['model_name']}", flush=True)
    source = resume_dir if resumed else config['model_name']
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSeq2SeqLM.from_pretrained(source).to(device)

    hashes = {key: sha256(config[key]) for key in ('train', 'validation')}
    if resumed and hashes != config['data_sha256']:
        raise ValueError('Dataset checksums changed since training')
    config['data_sha256'] = hashes
    train_rows = load_rows(config['train'], 'train')
    val_rows = load_rows(config['validation'], 'validation')
    check_disjoint(train_rows, val_rows)
    train_rows = sample_rows(train_rows, config['train_limit'], config['seed'])
    val_rows = sample_rows(val_rows, config['validation_limit'], config['seed'] + 1)
    train_data, val_data = T5QADataset(train_rows), T5QADataset(val_rows)
    del train_rows, val_rows
    collator = T5Collator(tokenizer, config['max_input_length'], config['max_target_length'])
    generator = torch.Generator().manual_seed(config['seed'])
    loader_options = dict(batch_size=config['batch_size'], collate_fn=collator, num_workers=0)
    train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_options)
    val_loader = DataLoader(val_data, shuffle=False, **loader_options)
    sample_indices = random.Random(config['seed']).sample(
        range(len(val_data)), min(config['eval_samples'], len(val_data)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'])
    updates_per_epoch = math.ceil(len(train_loader) / config['gradient_accumulation_steps'])
    total_updates = updates_per_epoch * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_updates * config['warmup_ratio']), total_updates)
    scaler = torch.amp.GradScaler(device.type, enabled=precision == 'fp16')
    start_epoch, best_loss, best_exact = 1, math.inf, -1.0
    stopping = EarlyStopping(config['early_stopping_patience'], config['early_stopping_min_delta'])
    if resumed:
        optimizer.load_state_dict(resumed['optimizer'])
        scheduler.load_state_dict(resumed['scheduler'])
        if resumed.get('scaler') and scaler.is_enabled():
            scaler.load_state_dict(resumed['scaler'])
        restore_rng(resumed['rng'])
        generator.set_state(resumed['loader_rng'])
        start_epoch = resumed['epoch'] + 1
        best_loss, best_exact = resumed['best_loss'], resumed['best_exact_match']
        stopping = EarlyStopping(config['early_stopping_patience'], config['early_stopping_min_delta'],
                                 resumed['early_stopping'])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir/'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    metrics_path = run_dir/'metrics.jsonl'
    print(f'Run: {run_dir}\nTraining: {len(train_data):,} | validation: {len(val_data):,} | '
          f'parameters: {sum(p.numel() for p in model.parameters()):,} | '
          f'effective batch: {config["batch_size"] * config["gradient_accumulation_steps"]}', flush=True)
    for epoch in range(start_epoch, epochs + 1):
        started = time.perf_counter()
        train_loss = epoch_loss(model, train_loader, device, precision, epoch, optimizer,
                                scheduler, scaler, config['gradient_accumulation_steps'],
                                args.log_every, args.progress)
        val_loss = epoch_loss(model, val_loader, device, precision, epoch,
                              log_every=args.log_every, progress_mode=args.progress)
        exact, examples = generated_metrics(
            model, val_data, tokenizer, collator, sample_indices, device, precision,
            config['batch_size'], config['max_target_length'], config['num_beams'])
        stop = stopping.update(val_loss)
        summary = {'event': 'epoch', 'epoch': epoch, 'train_loss': train_loss,
                   'validation_loss': val_loss, 'generated_exact_match': exact,
                   'generated_sample_count': len(sample_indices), 'early_stopping': stop,
                   'elapsed_seconds': time.perf_counter() - started}
        print(f'Epoch {epoch}/{epochs} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | '
              f'generated exact {exact:.2%} ({round(exact*len(sample_indices))}/{len(sample_indices)}) | '
              f'{summary["elapsed_seconds"]:.1f}s', flush=True)
        for example in examples[:3]:
            print(f"  {example['input']}\n  Expected: {example['expected']}\n"
                  f"  Predicted: {example['predicted']!r}", flush=True)
        state = {'format_version': 1, 'epoch': epoch, 'optimizer': optimizer.state_dict(),
                 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict() if scaler.is_enabled() else None,
                 'rng': rng_state(), 'loader_rng': generator.get_state(), 'best_loss': min(best_loss, val_loss),
                 'best_exact_match': max(best_exact, exact), 'early_stopping': stopping.state_dict()}
        # Best checkpoints are inference artifacts; only last needs bulky optimizer state.
        inference_state = {key: state[key] for key in
                           ('format_version', 'epoch', 'best_loss', 'best_exact_match')}
        if val_loss < best_loss:
            best_loss = val_loss
            inference_state['best_loss'] = best_loss
            save_checkpoint(run_dir/'best_loss', model, tokenizer, inference_state)
            print(f'Saved best-loss checkpoint: {run_dir/"best_loss"}', flush=True)
        if exact > best_exact:
            best_exact = exact
            state['best_exact_match'] = best_exact
            inference_state['best_exact_match'] = best_exact
            save_checkpoint(run_dir/'best_exact_match', model, tokenizer, inference_state)
            print(f'Saved best-exact checkpoint: {run_dir/"best_exact_match"}', flush=True)
        save_checkpoint(run_dir/'last', model, tokenizer, state)
        append_metric(metrics_path, summary)
        (run_dir/f'validation_epoch_{epoch:03d}.json').write_text(
            json.dumps(examples, ensure_ascii=False, indent=2), encoding='utf-8')
        if stop['should_stop']:
            print(f'Early stopping after epoch {epoch}.', flush=True)
            break
    return run_dir


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('train', 'validation', 'run-dir', 'resume'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--model-name')
    for name in ('batch-size', 'gradient-accumulation-steps', 'seed', 'max-input-length',
                 'max-target-length', 'train-limit', 'validation-limit', 'eval-samples',
                 'num-beams', 'early-stopping-patience', 'epochs'):
        parser.add_argument('--'+name, type=int)
    for name in ('learning-rate', 'weight-decay', 'warmup-ratio', 'early-stopping-min-delta'):
        parser.add_argument('--'+name, type=float)
    parser.add_argument('--precision', choices=['auto', 'off', 'fp16', 'bf16'])
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--log-every', type=int, default=50)
    parser.add_argument('--progress', choices=['auto', 'line', 'plain'], default='auto')
    return parser


def main():
    args = build_parser().parse_args()
    if args.threads < 1 or args.log_every < 1:
        raise SystemExit('--threads and --log-every must be positive')
    try:
        run(args)
    except KeyboardInterrupt:
        print('\nInterrupted. Resume the last directory to continue after the last completed epoch.')
        raise SystemExit(130)


if __name__ == '__main__':
    main()
