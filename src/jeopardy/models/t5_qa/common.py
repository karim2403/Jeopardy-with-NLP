"""Shared loading, scoring, precision, and checkpoint utilities."""
from contextlib import nullcontext
import json
from pathlib import Path
import random
import shutil

import torch

from ...evaluation import normalize_answer
from ..gru_seq2seq.evaluate import token_f1
from ..gru_seq2seq.model import select_device


def exact_match(prediction, target):
    expected = normalize_answer(target)
    return normalize_answer(prediction) == expected if expected else prediction.strip() == target.strip()


def resolve_precision(device, requested):
    if requested == 'auto':
        return 'bf16' if device.type == 'cuda' and torch.cuda.is_bf16_supported() else (
            'fp16' if device.type == 'cuda' else 'off')
    if requested != 'off' and device.type != 'cuda':
        raise ValueError(f'{requested} precision requires CUDA')
    return requested


def autocast_context(device, precision):
    if precision == 'off':
        return nullcontext()
    dtype = torch.bfloat16 if precision == 'bf16' else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def rng_state():
    return {'python': random.getstate(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def save_checkpoint(path, model, tokenizer, state):
    """Save a Hugging Face model plus optimizer/training state as one directory."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary, safe_serialization=True)
    tokenizer.save_pretrained(temporary)
    torch.save(state, temporary/'training_state.pt')
    if path.exists():
        shutil.rmtree(path)
    temporary.replace(path)


def load_training_state(path):
    return torch.load(Path(path)/'training_state.pt', map_location='cpu', weights_only=False)


def load_run_config(checkpoint):
    config_path = Path(checkpoint).parent/'config.json'
    if not config_path.exists():
        raise ValueError(f'Missing run configuration: {config_path}')
    return json.loads(config_path.read_text(encoding='utf-8'))


def model_device(requested):
    return select_device(requested)
