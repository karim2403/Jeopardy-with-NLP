"""Train/resume a small GRU with visible progress and portable checkpoints."""
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

from ...evaluation import normalize_answer
from ...tokenization import JeopardyTokenizer
from .checkpoint import atomic_save, load_checkpoint, restore_rng, rng_state, sha256
from .dataset import (QADataset, LengthBucketBatchSampler, LengthSortedBatchSampler,
                      check_disjoint, collate, load_rows, sample_rows)
from .model import GRUSeq2Seq, select_device
from .progress import ProgressLine
from .embedding_probes import DEFAULT_PAIRS, EmbeddingProbes, print_probes
from .early_stopping import EarlyStopping
from .precision import autocast_context, resolve_precision

ROOT = Path(__file__).resolve().parents[4]
DEFAULTS = dict(task="answer", embedding_dim=128, hidden_dim=256, batch_size=64,
                learning_rate=.001, seed=42, max_source_length=128, max_target_length=64,
                truncate=False, train_limit=None, validation_limit=None, eval_samples=128,
                early_stopping_patience=2, early_stopping_min_delta=0.0,
                precision='auto', bucket_multiplier=0)


def append_metric(path, entry):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False) + "\n")


def epoch_loss(model, loader, device, pad_id, optimizer=None, epoch=0, log_every=20,
               metrics_path=None, progress_mode='auto', precision='off', scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss = tokens = examples = 0
    recent_loss = recent_tokens = 0
    progress = ProgressLine(progress_mode)
    started = time.perf_counter()
    stage = "train" if training else "validation"
    with torch.set_grad_enabled(training):
        for step, batch in enumerate(loader, 1):
            source, target = batch['source'].to(device), batch['target'].to(device)
            labels = target[:, 1:]
            with autocast_context(device, precision):
                logits = model(source, batch['lengths'], target[:, :-1])
                count = labels.ne(pad_id).sum().item()
                summed = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                                         ignore_index=pad_id, reduction="sum")
            if not torch.isfinite(summed):
                raise RuntimeError("Non-finite loss; last completed-epoch checkpoint remains available")
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss = summed / count
                if scaler and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            batch_loss = summed.item()
            total_loss += batch_loss
            recent_loss += batch_loss
            recent_tokens += count
            tokens += count
            examples += source.size(0)
            if step == 1 or step % log_every == 0 or step == len(loader):
                elapsed = time.perf_counter() - started
                eta = elapsed / step * (len(loader) - step)
                progress.update(f"E{epoch} {stage} {step}/{len(loader)} | avg {total_loss/tokens:.4f} | "
                                f"recent {recent_loss/recent_tokens:.4f} | {examples/elapsed:.0f} ex/s | ETA {eta:.0f}s")
                if metrics_path:
                    append_metric(metrics_path, dict(event="progress", epoch=epoch, stage=stage, batch=step,
                                                    loss=total_loss/tokens, recent_loss=recent_loss/recent_tokens,
                                                    elapsed_seconds=elapsed))
                recent_loss = recent_tokens = 0
    progress.finish()
    return total_loss / tokens


@torch.inference_mode()
def generated_metrics(model, dataset, tokenizer, device, sample_indices, batch_size,
                      max_new_tokens, precision='off'):
    model.eval()
    results = []
    for offset in range(0, len(sample_indices), batch_size):
        items = [dataset[i] for i in sample_indices[offset:offset+batch_size]]
        batch = collate(items, tokenizer.pad_token_id)
        with autocast_context(device, precision):
            ids = model.generate(batch['source'].to(device), batch['lengths'], tokenizer.bos_token_id,
                                 tokenizer.eos_token_id, tokenizer.pad_token_id,
                                 tokenizer.unk_token_id, max_new_tokens)
        for item, output in zip(items, ids.cpu().tolist()):
            prediction = tokenizer.decode(output)
            gold = normalize_answer(item['target_text'])
            correct = normalize_answer(prediction) == gold if gold else prediction == item['target_text']
            results.append(dict(id=item['id'], task=item['task'], source=item['source_text'],
                                expected=item['target_text'], predicted=prediction, exact_match=correct,
                                reached_eos=tokenizer.eos_token_id in output))
    scores = {}
    for task in sorted({r['task'] for r in results}):
        group = [r for r in results if r['task'] == task]
        scores[task] = {"count": len(group), "exact_match": sum(r['exact_match'] for r in group)/len(group)}
    return scores, results


def run(args, model_class=GRUSeq2Seq, extra_defaults=None, extra_model_keys=(), model_type='gru_seq2seq'):
    # Windows redirected stdout may default to cp1252; generated text is Unicode.
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='backslashreplace')
    torch.set_num_threads(args.threads)
    device = select_device(args.device)
    # The requested policy is stored; auto resolves per execution device.
    resumed = load_checkpoint(args.resume) if args.resume else None
    defaults = {**DEFAULTS, **(extra_defaults or {})}
    if resumed:
        if Path(args.resume).name != "last.pt":
            raise ValueError("Resume from last.pt; best.pt is for prediction")
        config = resumed['config']
        saved_type = config.get('model_type', 'gru_seq2seq')
        if saved_type != model_type:
            raise ValueError(f'Checkpoint model type is {saved_type}, expected {model_type}')
        for key in (*defaults, 'train', 'validation', 'tokenizer', 'run_dir'):
            supplied = getattr(args, key)
            if supplied is not None and key in config and str(supplied) != str(config[key]):
                raise ValueError(f"Cannot change {key} when resuming; use a new run")
        for key, default in defaults.items():
            if key not in config:
                supplied = getattr(args, key)
                config[key] = default if supplied is None else supplied
        run_dir = Path(args.resume).resolve().parent
        if str(run_dir) != config['run_dir']:
            raise ValueError("Resume from the original run directory (inference supports moved runs)")
    else:
        config = {key: getattr(args, key) if getattr(args, key) is not None else value for key, value in defaults.items()}
        config.update(train=str((args.train or ROOT/'data/processed/train.jsonl').resolve()),
                      validation=str((args.validation or ROOT/'data/processed/validation.jsonl').resolve()),
                      tokenizer=str((args.tokenizer or ROOT/'artifacts/tokenizer').resolve()),
                      run_dir=str((args.run_dir or ROOT/'artifacts/models/gru_seq2seq/baseline').resolve()))
        run_dir = Path(config['run_dir'])
        if run_dir.exists() and any(run_dir.iterdir()):
            raise ValueError("Run directory is not empty; choose a new --run-dir or --resume its last.pt")
    epochs = args.epochs if args.epochs is not None else (config.get('epochs', 10))
    if epochs <= (resumed['epoch'] if resumed else 0):
        raise ValueError("--epochs is the total epoch count and must exceed the saved epoch")
    config['epochs'] = epochs
    config['model_type'] = model_type
    if getattr(args, 'no_probes', False):
        config['similarity_pairs'] = []
    elif getattr(args, 'probe_pair', None):
        config['similarity_pairs'] = args.probe_pair
    else:
        config.setdefault('similarity_pairs', DEFAULT_PAIRS)
    for key in ('embedding_dim', 'hidden_dim', 'batch_size', 'max_source_length', 'max_target_length', 'eval_samples'):
        if config[key] < 1:
            raise ValueError(f"{key} must be positive")
    for key in ('train_limit', 'validation_limit'):
        if config[key] is not None and config[key] < 1:
            raise ValueError(f"{key} must be positive")
    if config['learning_rate'] <= 0 or not math.isfinite(config['learning_rate']):
        raise ValueError("learning_rate must be positive and finite")
    if config['early_stopping_patience'] < 0 or config['early_stopping_min_delta'] < 0:
        raise ValueError('Early-stopping settings must be nonnegative')
    if config['bucket_multiplier'] < 0:
        raise ValueError('bucket_multiplier must be nonnegative')
    resolved_precision = resolve_precision(device, config['precision'])
    random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.benchmark = False
    print(f"Device: {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == 'cuda' else '') +
          f" | precision: {resolved_precision}", flush=True)
    print(f"PyTorch {torch.__version__} | run: {run_dir}\nSettings: {json.dumps(config, indent=2)}", flush=True)
    tokenizer = JeopardyTokenizer.load(run_dir/'tokenizer' if resumed else config['tokenizer'])
    hashes = {key: sha256(config[key]) for key in ('train', 'validation')}
    if resumed and hashes != config['data_sha256']:
        raise ValueError("Dataset checksums changed since the checkpoint")
    config['data_sha256'] = hashes
    if resumed and sha256(run_dir/'tokenizer/tokenizer.json') != config['tokenizer_sha256']:
        raise ValueError("Tokenizer changed since checkpoint")
    train_rows, val_rows = load_rows(config['train'], 'train'), load_rows(config['validation'], 'validation')
    check_disjoint(train_rows, val_rows)
    train_rows = sample_rows(train_rows, config['train_limit'], config['seed'])
    val_rows = sample_rows(val_rows, config['validation_limit'], config['seed'] + 1)
    data_options = {key: config[key] for key in ('task', 'max_source_length', 'max_target_length', 'truncate')}
    train_data, val_data = QADataset(train_rows, tokenizer, **data_options), QADataset(val_rows, tokenizer, **data_options)
    print(f"Training data: {train_data.stats}\nValidation data: {val_data.stats}", flush=True)
    del train_rows, val_rows
    config['model'] = dict(vocab_size=tokenizer.vocab_size, pad_token_id=tokenizer.pad_token_id,
                           embedding_dim=config['embedding_dim'], hidden_dim=config['hidden_dim'],
                           **{key: config[key] for key in extra_model_keys})
    model = model_class(**config['model']).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    scaler = torch.amp.GradScaler(device.type, enabled=resolved_precision == 'fp16')
    generator = torch.Generator().manual_seed(config['seed'])
    collator = partial(collate, pad_id=tokenizer.pad_token_id)
    if config['bucket_multiplier']:
        train_loader = DataLoader(train_data, batch_sampler=LengthBucketBatchSampler(
            train_data, config['batch_size'], generator, config['bucket_multiplier']),
            collate_fn=collator, num_workers=0)
        val_loader = DataLoader(val_data, batch_sampler=LengthSortedBatchSampler(
            val_data, config['batch_size']), collate_fn=collator, num_workers=0)
    else:
        loader_options = dict(batch_size=config['batch_size'], collate_fn=collator, num_workers=0)
        train_loader = DataLoader(train_data, shuffle=True, generator=generator, **loader_options)
        val_loader = DataLoader(val_data, shuffle=False, **loader_options)
    sample_indices = random.Random(config['seed']).sample(range(len(val_data)), min(config['eval_samples'], len(val_data)))
    best_loss, start_epoch = math.inf, 1
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir/'metrics.jsonl'
    if resumed:
        model.load_state_dict(resumed['model'])
        optimizer.load_state_dict(resumed['optimizer'])
        if resumed.get('scaler') and scaler.is_enabled():
            scaler.load_state_dict(resumed['scaler'])
        restore_rng(resumed['rng'], generator)
        best_loss, start_epoch = resumed['best_val_loss'], resumed['epoch'] + 1
        # Remove interrupted-epoch progress before replaying that epoch.
        if metrics_path.exists():
            kept = []
            for line in metrics_path.read_text(encoding='utf-8').splitlines():
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry['epoch'] <= resumed['epoch']:
                    kept.append(line)
            metrics_path.write_text('\n'.join(kept) + '\n', encoding='utf-8')
        print(f"Resuming after epoch {resumed['epoch']}; target total epochs: {epochs}", flush=True)
    else:
        tokenizer.save(run_dir/'tokenizer')
    config['tokenizer_sha256'] = sha256(run_dir/'tokenizer/tokenizer.json')
    config['torch_version'] = str(torch.__version__)
    (run_dir/'config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,} | vocab: {tokenizer.vocab_size:,} | tokenizer SHA256: {config['tokenizer_sha256']}", flush=True)
    probes = EmbeddingProbes(tokenizer, config['similarity_pairs'], resumed.get('probe_state') if resumed else None)
    stopping = EarlyStopping(config['early_stopping_patience'], config['early_stopping_min_delta'],
                             resumed.get('early_stopping') if resumed else None)
    if probes.pairs:
        initial_probes = probes.measure(model.embedding.weight, start_epoch - 1)
        print_probes(initial_probes, show_pieces=True)
        append_metric(metrics_path, dict(event='embedding_probes', **initial_probes))
    progress_mode = getattr(args, 'progress', 'auto')
    for epoch in range(start_epoch, epochs + 1):
        started = time.perf_counter()
        train_loss = epoch_loss(model, train_loader, device, tokenizer.pad_token_id, optimizer,
                                epoch, args.log_every, metrics_path, progress_mode,
                                resolved_precision, scaler)
        val_loss = epoch_loss(model, val_loader, device, tokenizer.pad_token_id, epoch=epoch,
                              log_every=args.log_every, metrics_path=metrics_path,
                              progress_mode=progress_mode, precision=resolved_precision)
        scores, examples = generated_metrics(model, val_data, tokenizer, device, sample_indices,
                                             config['batch_size'], config['max_target_length'] - 1,
                                             resolved_precision)
        stop_status = stopping.update(val_loss)
        summary = dict(event='epoch', epoch=epoch, train_loss=train_loss, validation_loss=val_loss,
                       generated=scores, early_stopping=stop_status,
                       elapsed_seconds=time.perf_counter()-started)
        print(f"Epoch {epoch}/{epochs} complete | train loss {train_loss:.4f} | val loss {val_loss:.4f} | "
              f"generated exact match {scores} | {summary['elapsed_seconds']:.1f}s", flush=True)
        if probes.pairs:
            probe_report = probes.measure(model.embedding.weight, epoch)
            summary['embedding_probes'] = probe_report
            print_probes(probe_report)
        for example in examples[:3]:
            print(f"  [{example['task']}] {example['source']}\n  Expected: {example['expected']}\n  Predicted: {example['predicted']!r}", flush=True)
        improved = val_loss < best_loss
        best_loss = min(best_loss, val_loss)
        payload = dict(format_version=1, epoch=epoch, best_val_loss=best_loss, config=config,
                       model=model.state_dict(), optimizer=optimizer.state_dict(), rng=rng_state(generator),
                       probe_state=probes.state, early_stopping=stopping.state_dict(),
                       scaler=scaler.state_dict() if scaler.is_enabled() else None)
        if improved:
            atomic_save(payload, run_dir/'best.pt')
            print(f"Saved best checkpoint: {run_dir/'best.pt'}", flush=True)
        atomic_save(payload, run_dir/'last.pt')
        append_metric(metrics_path, summary)
        (run_dir/f'validation_epoch_{epoch:03d}.json').write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"Saved resume checkpoint: {run_dir/'last.pt'}", flush=True)
        if stop_status['should_stop']:
            print(f"Early stopping after epoch {epoch}: validation loss did not improve by at least "
                  f"{config['early_stopping_min_delta']:g} for {config['early_stopping_patience']} consecutive epochs. "
                  f"Best checkpoint: {run_dir/'best.pt'}", flush=True)
            break
    return run_dir


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('train', 'validation', 'tokenizer', 'run-dir', 'resume'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--task', choices=['answer', 'clue', 'joint'])
    for name in ('embedding-dim', 'hidden-dim', 'batch-size', 'seed', 'max-source-length',
                 'max-target-length', 'train-limit', 'validation-limit', 'eval-samples', 'epochs'):
        parser.add_argument('--'+name, type=int)
    parser.add_argument('--learning-rate', type=float)
    parser.add_argument('--precision', choices=['auto', 'off', 'fp16', 'bf16'],
                        help='CUDA mixed precision; auto prefers bf16 when supported')
    parser.add_argument('--bucket-multiplier', type=int,
                        help='Length-bucket pool size in batches; 0 disables bucketing')
    parser.add_argument('--early-stopping-patience', type=int,
                        help='Stop after this many non-improving validation epochs; 0 disables')
    parser.add_argument('--early-stopping-min-delta', type=float,
                        help='Minimum validation-loss decrease counted as an early-stopping improvement')
    parser.add_argument('--truncate', action='store_true', default=None)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--log-every', type=int, default=20)
    parser.add_argument('--progress', choices=['auto', 'line', 'plain'], default='auto',
                        help='auto replaces progress on terminals; plain prints separate lines')
    probes = parser.add_mutually_exclusive_group()
    probes.add_argument('--probe-pair', nargs=2, action='append', metavar=('LEFT', 'RIGHT'),
                        help='Repeat to override the default embedding-similarity pairs')
    probes.add_argument('--no-probes', action='store_true', help='Disable embedding-similarity diagnostics')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.threads < 1 or args.log_every < 1:
        parser.error('--threads and --log-every must be positive')
    try:
        run(args)
    except KeyboardInterrupt:
        print('\nInterrupted. Resume last.pt to restart after the last completed epoch.', flush=True)
        raise SystemExit(130)


if __name__ == '__main__':
    main()
