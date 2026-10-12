"""Native DeepSpeed safety checkpoints, exact RNG restoration and update recovery."""
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist


class PackedUpdateView:
    """Keep complete original update records; skip before feature fetching."""
    def __init__(self, sampler, first_update):
        if not 1 <= first_update <= sampler.total_updates:
            raise ValueError('Invalid original packed update index')
        self.sampler, self.first_update = sampler, first_update

    @property
    def epoch(self):
        return self.sampler.epoch

    @epoch.setter
    def epoch(self, value):
        self.sampler.epoch = value

    def __getattr__(self, name):
        return getattr(self.sampler, name)

    def __iter__(self):
        for batch in self.sampler:
            if batch[0][1] >= self.first_update:
                yield batch

    def __len__(self):
        return sum(item['update'] >= self.first_update for item in self.sampler.plan())


def capture_rng():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state())


def restore_rng(value):
    random.setstate(value['python']); np.random.set_state(value['numpy'])
    torch.set_rng_state(value['torch'].cpu()); torch.cuda.set_rng_state(value['cuda'].cpu())


def save_json(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(temporary, path)


def completed_tags(directory):
    return sorted(p for p in Path(directory).glob('step*') if p.is_dir()
                  and re.fullmatch(r'step\d{6}', p.name) and (p/'complete_manifest.json').is_file())


def retain_latest_two(directory):
    """Called only after all-rank save/barrier and durable complete manifest."""
    directory = Path(directory).resolve()
    for tag in completed_tags(directory)[:-2]:
        if tag.is_symlink() or tag.parent.resolve() != directory:
            raise ValueError('Refuse to prune checkpoint outside this run')
        shutil.rmtree(tag)


def save_safety_checkpoint(engine, directory, *, rank, world, step, client_state):
    began = time.monotonic()
    directory = Path(directory); tag = f'step{step:06d}'
    if (directory/tag).exists():
        raise FileExistsError('Never overwrite a completed or partial safety checkpoint')
    rng = capture_rng()
    states = [None]*world
    if world > 1:
        dist.all_gather_object(states, rng)
    else:
        states[0] = rng
    state = dict(client_state, rng_state_by_rank=states,
                 safety_checkpoint=True, saved_step=step)
    result = engine.save_checkpoint(str(directory), tag=tag, client_state=state)
    if result is False:
        raise RuntimeError('DeepSpeed failed to save safety checkpoint')
    if world > 1:
        dist.barrier()
    report = None
    if rank == 0:
        files = {str(p.relative_to(directory/tag)): p.stat().st_size
                 for p in (directory/tag).rglob('*') if p.is_file()}
        optimizer_files = [name for name in files if name.endswith('optim_states.pt')]
        if not files or len(optimizer_files) != world or not any(name.endswith('model_states.pt') for name in files):
            raise RuntimeError('Incomplete all-rank DeepSpeed checkpoint inventory')
        report = dict(complete=True, step=step, world_size=world, files=files,
                      bytes=sum(files.values()), save_seconds=time.monotonic()-began,
                      next_epoch=client_state['next_epoch'], next_update=client_state['next_update'])
        save_json(report, directory/tag/'complete_manifest.json')
        retain_latest_two(directory)
        save_json(dict(report, directory=str(directory), tag=tag), directory.parent/'safety_checkpoint.json')
    if world > 1:
        values = [report]
        dist.broadcast_object_list(values, src=0)
        report = values[0]
    restore_rng(rng)
    return report


def verify_checkpoint_inventory(directory, tag=None):
    directory = Path(directory)
    candidates=completed_tags(directory)
    if tag is None:
        if not candidates:
            raise ValueError('No completed safety checkpoint is available')
        tag=candidates[-1].name
    if not re.fullmatch(r'step\d{6}', tag):
        raise ValueError('Unknown safety checkpoint tag')
    report = json.loads((directory/tag/'complete_manifest.json').read_text())
    if not report.get('complete'):
        raise ValueError('Safety checkpoint lacks a completed all-rank manifest')
    for name, size in report['files'].items():
        path = directory/tag/name
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f'Safety checkpoint changed or incomplete: {name}')
    return dict(report,tag=tag)


def archive_replayed_progress(out, *, saved_step, rank, world):
    """A resume keeps evidence but prevents duplicate steps or partial-tag reuse."""
    from datetime import datetime, timezone
    out=Path(out)
    identity=[datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') if rank==0 else None]
    if world>1:
        dist.broadcast_object_list(identity,src=0)
    archive=out/'resume_archives'/identity[0]
    log=out/f'train_rank{rank}.jsonl'
    if saved_step>0 and not log.exists():
        raise ValueError('Resume requires the original per-rank training log')
    if log.exists():
        lines=log.read_text().splitlines()
        rows=[json.loads(line) for line in lines if line.strip()]
        if [row['step'] for row in rows if row['step']<=saved_step]!=list(range(1,saved_step+1)):
            raise ValueError('Resume logs do not fully cover the saved checkpoint progress')
        if any(row['step']>saved_step for row in rows):
            archive.mkdir(parents=True,exist_ok=True)
            shutil.copy2(log,archive/log.name)
            temporary=log.with_suffix('.resume.tmp')
            temporary.write_text(''.join(json.dumps(row)+'\n' for row in rows if row['step']<=saved_step))
            os.replace(temporary,log)
    if rank==0:
        # A failed final export/completion write is retried from the exact
        # DeepSpeed state. Preserve the previous attempt before publishing anew.
        for name in ('hf_checkpoint', 'hf_checkpoint.incomplete', 'completion.json'):
            path = out/name
            if path.exists():
                archive.mkdir(parents=True, exist_ok=True)
                path.rename(archive/name)
        checkpoints=out/'safety_deepspeed'
        for path in checkpoints.glob('step*'):
            if (re.fullmatch(r'step\d{6}',path.name) and path.is_dir() and
                    (not (path/'complete_manifest.json').exists() or int(path.name[4:])>saved_step)):
                target=archive/'safety_deepspeed'/path.name
                target.parent.mkdir(parents=True,exist_ok=True)
                path.rename(target)
    if world>1:
        dist.barrier()
    return archive


def training_offsets(state):
    offsets = state.get('timing_offsets', {})
    return {key: float(offsets.get(key, 0.)) for key in
            ['elapsed', 'training_seconds', 'benchmark_seconds']}


def state_fingerprint(value):
    """Exact bounded-memory fingerprint, including native ZeRO optimizer shards."""
    digest = hashlib.sha256()
    def visit(item):
        if torch.is_tensor(item):
            digest.update(str((tuple(item.shape), item.dtype)).encode())
            flat = item.detach().reshape(-1)
            elements = max(1, 16*1024*1024//max(1,item.element_size()))
            for start in range(0, flat.numel(), elements):
                data = flat[start:start+elements].cpu().contiguous().view(torch.uint8).numpy()
                digest.update(data.tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                digest.update(repr(key).encode());visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for part in item:visit(part)
        else:
            digest.update(repr(item).encode())
    visit(value)
    return digest.hexdigest()
