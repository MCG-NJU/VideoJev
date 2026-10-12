"""Bounded-memory loading of one HF submodule from the official safetensors.

Unlike Transformers' current ZeRO-3 loader, this never builds a state dict
holding every checkpoint shard.  All ranks construct sharded parameters;
only rank zero reads a tensor before broadcasting its initialized partition.
The ordinary path is also usable for the small replicated frozen ViT.
"""
from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path

import torch


def _weight_map(model_path):
    from safetensors import safe_open
    path = Path(model_path)
    index = path / 'model.safetensors.index.json'
    if index.exists():
        with index.open() as handle:
            mapping = json.load(handle)['weight_map']
        return {name: path / shard for name, shard in mapping.items()}
    single = path / 'model.safetensors'
    if not single.is_file():
        raise FileNotFoundError(f'No HuggingFace safetensors checkpoint found in {path}')
    with safe_open(single, framework='pt', device='cpu') as handle:
        return {name: single for name in handle.keys()}


def load_hf_submodule_weights(model, model_path, *, prefix, zero3=False):
    """Load ``prefix + parameter_name`` without materializing the full model.

    ``model`` must already exist, under ``deepspeed.zero.Init(dtype=float32)``
    for the ZeRO-3 path.  Parameter keys/shapes must match the official fused
    Qwen3.5-MoE format used by Qwen3.6; mismatches fail rather than silently
    initializing missing trainable weights.  There is no forward/reload check.
    """
    from safetensors import safe_open
    mapping = _weight_map(model_path)
    parameters = dict(model.named_parameters())
    persistent_state = set(model.state_dict())
    buffers = {name: value for name, value in model.named_buffers() if name in persistent_state}
    expected = {prefix + name for name in parameters} | {prefix + name for name in buffers}
    absent = expected - mapping.keys()
    if absent:
        raise RuntimeError(f'Missing checkpoint tensors for {prefix}: {sorted(absent)[:10]}')
    rank = torch.distributed.get_rank() if zero3 else 0
    if zero3:
        import deepspeed
        if any(not hasattr(param, 'ds_id') for param in parameters.values()):
            raise ValueError('Construct the language submodule inside deepspeed.zero.Init before zero3 loading')
    loaded = 0
    with ExitStack() as stack, torch.no_grad():
        readers = {}

        def read(name):
            path = mapping[prefix + name]
            if path not in readers:
                readers[path] = stack.enter_context(safe_open(path, framework='pt', device='cpu'))
            return readers[path].get_tensor(prefix + name)

        for name, param in parameters.items():
            if zero3:
                with deepspeed.zero.GatheredParameters([param], modifier_rank=0):
                    if rank == 0:
                        source = read(name)
                        if source.shape != param.shape:
                            raise ValueError(f'Checkpoint shape mismatch at {prefix + name}: {source.shape} vs {param.shape}')
                        param.copy_(source)
                        del source
            else:
                source = read(name)
                if source.shape != param.shape:
                    raise ValueError(f'Checkpoint shape mismatch at {prefix + name}: {source.shape} vs {param.shape}')
                param.copy_(source)
                del source
            param._is_hf_initialized = True
            loaded += 1
        # Buffers are not partitioned by ZeRO-3; each rank loads these few small
        # persistent tensors directly. Nonpersistent RoPE buffers stay native.
        for name, buffer in buffers.items():
            source = read(name)
            if source.shape != buffer.shape:
                raise ValueError(f'Checkpoint buffer shape mismatch at {prefix + name}')
            buffer.copy_(source)
            del source
            loaded += 1
    return dict(loaded_tensors=loaded, prefix=prefix, zero3=zero3,
                loader='streaming_safetensors_per_parameter')
