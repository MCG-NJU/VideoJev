"""Bounded-memory export and completed-only ZeRO-3 resume for the MoE run."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil

import numpy as np
import torch
import torch.distributed as dist


FORMAT = 'timejev_qwen36_task_v1'


def _rank_world():
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def _rank0_call(fn):
    rank, world = _rank_world()
    payload = [None]
    if rank == 0:
        try:
            payload[0] = {'result': fn()}
        except Exception as exc:
            payload[0] = {'error': f'{type(exc).__name__}: {exc}'}
    if world > 1:
        dist.broadcast_object_list(payload, src=0)
    if 'error' in payload[0]:
        raise RuntimeError(payload[0]['error'])
    return payload[0]['result']


def _json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2, default=str)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def _restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'].cpu())


def _remove_local_directory(path, parent):
    path, parent = Path(path), Path(parent)
    if path.is_symlink() or path.parent.resolve() != parent.resolve():
        raise ValueError('Refuse to delete a checkpoint path outside this run')
    if path.exists():
        shutil.rmtree(path)


def _complete_manifest(path):
    path = Path(path)
    try:
        report = json.loads((path / 'complete_manifest.json').read_text())
        if report.get('complete') is not True:
            return None
        for name, size in report['files'].items():
            file = path / name
            if not file.is_file() or file.stat().st_size != size:
                return None
        return report
    except (OSError, ValueError, KeyError, TypeError):
        return None


def find_latest_complete_checkpoint(directory):
    """Return (root, tag, manifest); ignore interrupted saves and stale latest."""
    directory = Path(directory)
    paths = sorted((p for p in directory.glob('step*')
                    if p.is_dir() and re.fullmatch(r'step\d{6,}', p.name)),
                   key=lambda p: int(p.name[4:]), reverse=True)
    for path in paths:
        report = _complete_manifest(path)
        if report is not None:
            return directory, path.name, report
    return None


def save_checkpoint(engine, directory, step, client_state, *, keep=2):
    """Save all-rank optimizer/RNG/progress, excluding the frozen backbone.

    ``client_state`` must carry the caller's sampler/epoch/update position and
    data/config identity. DeepSpeed saves optimizer/scheduler progress itself.
    Only fully written checkpoints are eligible for resume or retention.
    """
    if keep < 1 or int(step) < 0:
        raise ValueError('Checkpoint retention and step must be valid')
    directory, tag = Path(directory), f'step{int(step):06d}'
    target = directory / tag
    rank, world = _rank_world()
    rng = _capture_rng()
    states = [None] * world
    if world > 1:
        dist.all_gather_object(states, rng)
    else:
        states[0] = rng

    def prepare():
        directory.mkdir(parents=True, exist_ok=True)
        owner_file = directory / '.timejev_qwen36_owner.json'
        owner = {'format': 'timejev_qwen36_resume_v1', 'run': str(directory.parent.resolve())}
        if owner_file.exists() and json.loads(owner_file.read_text()) != owner:
            raise ValueError('Resume checkpoint directory belongs to a different run')
        if not owner_file.exists():
            _json(owner_file, owner)
        existing = _complete_manifest(target)
        if existing is not None:
            return True
        _remove_local_directory(target, directory)
        return False

    already_complete = _rank0_call(prepare)
    if already_complete:
        return target
    state = dict(client_state, rng_state_by_rank=states, saved_step=int(step),
                 saved_world_size=world, exclude_frozen_parameters=True,
                 optimizer_lrs=[float(group['lr']) for group in engine.optimizer.param_groups])
    try:
        result = engine.save_checkpoint(str(directory), tag=tag, client_state=state,
                                        save_latest=False, exclude_frozen_parameters=True)
        if result is False:
            raise RuntimeError('DeepSpeed checkpoint save returned failure')
        if world > 1:
            dist.barrier()

        def finish():
            files = {str(p.relative_to(target)): p.stat().st_size
                     for p in target.rglob('*') if p.is_file()}
            optimizer_files = [name for name in files if name.endswith('optim_states.pt')]
            if len(optimizer_files) != world or not any(name.endswith('model_states.pt') for name in files):
                raise RuntimeError('Missing all-rank ZeRO-3 checkpoint shards')
            report = dict(complete=True, kind='deepspeed_resume', step=int(step),
                world_size=world, files=files, bytes=sum(files.values()),
                exclude_frozen_parameters=True, client_state_keys=sorted(state),
                next_epoch=state.get('next_epoch'), next_update=state.get('next_update'),
                created_utc=datetime.now(timezone.utc).isoformat())
            _json(target / 'complete_manifest.json', report)
            temporary = directory / 'latest.tmp'
            temporary.write_text(tag + '\n')
            os.replace(temporary, directory / 'latest')
            completed = sorted((p for p in directory.glob('step*')
                if p.is_dir() and re.fullmatch(r'step\d{6,}', p.name)
                and _complete_manifest(p) is not None), key=lambda p: int(p.name[4:]))
            for path in completed[:-keep]:
                _remove_local_directory(path, directory)
            return True
        _rank0_call(finish)
    finally:
        _restore_rng(rng)
    return target


def load_checkpoint(engine, directory, *, tag=None):
    """Resume only a complete save; restore this rank's saved RNG state."""
    directory = Path(directory)

    def choose():
        selected = find_latest_complete_checkpoint(directory)
        if tag is not None:
            if not re.fullmatch(r'step\d{6,}', tag):
                raise ValueError('Invalid checkpoint tag')
            report = _complete_manifest(directory / tag)
            selected = (directory, tag, report) if report is not None else None
        if selected is None:
            return None
        return {'tag': selected[1], 'manifest': selected[2]}

    selected = _rank0_call(choose)
    if selected is None:
        return None
    rank, world = _rank_world()
    if selected['manifest']['world_size'] != world:
        raise ValueError('Exact optimizer/data/RNG resume requires the original world size')
    path, state = engine.load_checkpoint(str(directory), tag=selected['tag'],
        load_module_strict=False, load_optimizer_states=True, load_lr_scheduler_states=True)
    if path is None or state is None:
        raise RuntimeError('DeepSpeed could not load the selected complete checkpoint')
    if not state.get('exclude_frozen_parameters'):
        raise ValueError('Not a Qwen3.6 frozen-parameter-excluding checkpoint')
    if int(state['saved_step']) != int(selected['manifest']['step']):
        raise ValueError('Checkpoint client progress and completion manifest disagree')
    _restore_rng(state['rng_state_by_rank'][rank])
    return state


def export_checkpoint(model, directory, config, checkpoint_id, step, *,
                      max_shard_bytes=2 * 1024**3, head_only=False):
    """Stream exact FP32 task weights; never collect a 140 GB CPU state dict.

    Full export uses the task wrapper's original keys (language_model.* and
    head.*). It is intentionally not an unmodified HF conditional-generation
    model: the task skips the last MoE block and has its own decision head.
    """
    from safetensors.torch import save_file
    directory = Path(directory)
    staging = directory.with_name(directory.name + '.incomplete')
    rank, world = _rank_world()
    if max_shard_bytes < 1:
        raise ValueError('Shard byte limit must be positive')

    def prepare():
        directory.parent.mkdir(parents=True, exist_ok=True)
        existing = _complete_manifest(directory)
        if existing is not None:
            if (existing.get('checkpoint_id') != checkpoint_id or existing.get('step') != int(step)
                    or existing.get('head_only', False) != bool(head_only)):
                raise FileExistsError('Export directory already contains a different completed checkpoint')
            return True
        if directory.exists():
            raise FileExistsError('Refuse to overwrite an unrecognized export directory')
        _remove_local_directory(staging, directory.parent)
        staging.mkdir()
        return False
    if _rank0_call(prepare):
        return directory

    pending, pending_bytes, shard_number, total_bytes = {}, 0, 0, 0
    weight_map, shapes = {}, {}

    def flush():
        nonlocal pending, pending_bytes, shard_number
        if pending_bytes == 0:
            return
        shard_number += 1
        filename = f'model-{shard_number:05d}.safetensors'
        def write():
            save_file(pending, staging / filename, metadata={'format': 'pt'})
            for name in pending:
                weight_map[name] = filename
            return True
        _rank0_call(write)
        pending, pending_bytes = {}, 0

    for name, param in model.named_parameters():
        if head_only and not name.startswith('head.'):
            continue
        if not (name.startswith('language_model.') or name.startswith('head.')):
            raise ValueError(f'Unexpected registered parameter in task export: {name}')
        shape = tuple(getattr(param, 'ds_shape', param.shape))
        size = int(np.prod(shape, dtype=np.int64)) * 4
        if pending_bytes and pending_bytes + size > max_shard_bytes:
            flush()
        if hasattr(param, 'ds_id'):
            import deepspeed
            with deepspeed.zero.GatheredParameters([param]):
                if rank == 0:
                    pending[name] = param.detach().to(device='cpu', dtype=torch.float32, copy=True).contiguous()
        elif rank == 0:
            pending[name] = param.detach().to(device='cpu', dtype=torch.float32, copy=True).contiguous()
        shapes[name] = list(shape)
        pending_bytes += size
        total_bytes += size
    # Preserve only persistent model buffers; ordinary Qwen RoPE buffers are
    # nonpersistent and reconstructed from the recorded native configuration.
    persistent_keys = set(model.state_dict())
    for name, buffer in model.named_buffers():
        if name not in persistent_keys or (head_only and not name.startswith('head.')):
            continue
        size = buffer.numel() * buffer.element_size()
        if pending_bytes and pending_bytes + size > max_shard_bytes:
            flush()
        if rank == 0:
            pending[name] = buffer.detach().to(device='cpu', copy=True).contiguous()
        shapes[name] = list(buffer.shape)
        pending_bytes += size
        total_bytes += size
    flush()

    def finish():
        if not weight_map:
            raise ValueError('Task export contains no tensors')
        recipe = dict(config)
        native = getattr(model, 'native_config', None)
        native = native.to_dict() if hasattr(native, 'to_dict') else native
        base = recipe.get('model_path', recipe.get('base', recipe.get('base_model')))
        base_config_sha256 = None
        if base and (Path(base) / 'config.json').is_file():
            base_config_sha256 = hashlib.sha256((Path(base) / 'config.json').read_bytes()).hexdigest()
        decision_metadata = getattr(model, 'decision_metadata', None)
        readout_metadata = (getattr(model, 'readout_metadata', None)
                            or (decision_metadata or {}).get('readout'))
        metadata = dict(format=FORMAT, checkpoint_id=checkpoint_id, step=int(step),
            head_only=bool(head_only), model_class=type(model).__name__, recipe=recipe,
            decision_metadata=decision_metadata, readout_metadata=readout_metadata,
            parameter_dtype='float32', tensor_shapes=shapes,
            vision_reference=dict(base_model_path=str(base) if base else None,
                revision=recipe.get('model_revision', recipe.get('revision')),
                config_sha256=base_config_sha256, frozen=True, copied_weights=False),
            loader='timejev.qwen36_checkpoint.load_exported_task_checkpoint',
            architecture_note='Task wrapper checkpoint; restore trimmed final MoE readout before loading.')
        _json(staging / 'model.safetensors.index.json',
              {'metadata': {'total_size': total_bytes}, 'weight_map': weight_map})
        _json(staging / 'task_model_config.json', metadata)
        if native is not None:
            _json(staging / 'native_config.json', native)
        files = {p.name: p.stat().st_size for p in staging.iterdir() if p.is_file()}
        _json(staging / 'complete_manifest.json', dict(complete=True, kind='task_export',
            format=FORMAT, checkpoint_id=checkpoint_id, step=int(step), head_only=bool(head_only),
            files=files, bytes=sum(files.values()), tensor_bytes=total_bytes))
        os.replace(staging, directory)
        return True
    _rank0_call(finish)
    return directory


def load_exported_task_checkpoint(model, directory, *, zero3=False):
    """Load into an already constructed task model, including its trimmed readout."""
    from .qwen36_load import load_hf_submodule_weights
    directory = Path(directory)
    report = _complete_manifest(directory)
    if report is None or report.get('format') != FORMAT:
        raise ValueError('No complete Qwen3.6 task export is available')
    metadata = json.loads((directory / 'task_model_config.json').read_text())
    if metadata['format'] != FORMAT:
        raise ValueError('Wrong task checkpoint format')
    expected_decision = getattr(model, 'decision_metadata', None)
    if expected_decision is not None and metadata.get('decision_metadata') != expected_decision:
        raise ValueError('Checkpoint token/readout/head decision metadata differs from the target model')
    head_only = metadata.get('head_only', False)
    module = model.head if head_only else model
    result = load_hf_submodule_weights(module, directory,
        prefix='head.' if head_only else '', zero3=zero3)
    return dict(metadata=metadata, loading=result)
