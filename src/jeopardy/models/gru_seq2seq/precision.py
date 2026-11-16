"""Automatic mixed-precision policy shared by model variants."""
from contextlib import nullcontext

import torch


def resolve_precision(device, requested='auto'):
    if requested not in ('auto', 'off', 'fp16', 'bf16'):
        raise ValueError('precision must be auto, off, fp16, or bf16')
    if requested == 'auto':
        if device.type != 'cuda':
            return 'off'
        return 'bf16' if torch.cuda.is_bf16_supported() else 'fp16'
    if requested != 'off' and device.type != 'cuda':
        raise RuntimeError(f'{requested} mixed precision requires CUDA')
    if requested == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('This CUDA device does not support bf16')
    return requested


def autocast_context(device, precision):
    if precision == 'off':
        return nullcontext()
    dtype = torch.bfloat16 if precision == 'bf16' else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)
