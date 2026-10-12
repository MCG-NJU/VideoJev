import contextlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler


def setup(seed):
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local)
    torch.set_num_threads(2)
    if world > 1:
        dist.init_process_group('nccl', device_id=torch.device('cuda', local))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    return rank, world, torch.device('cuda', local)


class BucketSampler(Sampler):
    """Length bucketing with matched global steps and deterministic DDP padding."""
    def __init__(self, lengths, batch, rank=0, world=1, seed=0, reference_world=None):
        self.lengths, self.batch, self.rank, self.world, self.seed = lengths, batch, rank, world, seed
        self.reference_world = reference_world or world
        if self.reference_world % world:
            raise ValueError('Reference world must be divisible by physical world')
        self.epoch = 0

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        order = list(range(len(self.lengths)))
        rng.shuffle(order)
        global_batch = self.batch * self.reference_world
        pad = (-len(order)) % global_batch
        if not order:
            raise ValueError('Empty training dataset')
        order += (order * math.ceil(pad / len(order)))[:pad]
        groups = []
        for start in range(0, len(order), global_batch * 32):
            bucket = sorted(order[start:start + global_batch * 32], key=lambda i: self.lengths[i])
            groups.extend(bucket[j:j + global_batch] for j in range(0, len(bucket), global_batch))
        rng.shuffle(groups)
        for group in groups:
            # Execute the original reference ranks sequentially on each GPU.
            # Increasing accumulation by the same ratio preserves each global
            # optimizer batch, its order, and its exact duplicate padding.
            for virtual_rank in range(self.rank, self.reference_world, self.world):
                yield group[virtual_rank::self.reference_world]

    def __len__(self):
        return math.ceil(len(self.lengths) / (self.batch * self.reference_world)) * (self.reference_world // self.world)


@torch.inference_mode()
def predict(model, loader, device):
    from .data import to_device
    model.eval()
    results = []
    cache = model.video_cache(max_tokens=147456) if hasattr(model, 'video_cache') else contextlib.nullcontext()
    with cache:
        for batch in loader:
            device_batch = to_device(batch, device)
            device_batch.pop('labels', None)
            output = model(**device_batch)
            logits = output['logits'].float().cpu()
            for i, (row, times) in enumerate(zip(batch['rows'], batch['times'])):
                results.append(dict(id=row['id'], times=times.tolist(),
                                    logits=logits[i, :len(times)].tolist(),
                                    bounds=batch['audits'][i]['score_bounds'],
                                    sampled_duration=row['sampled_duration'], audit=batch['audits'][i]))
    return results


def atomic_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    os.replace(temporary, path)


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def make_loader(dataset, collator, workers, **kwargs):
    from torch.utils.data import DataLoader
    # Spawn avoids inheriting initialized CUDA state; one queued video per worker
    # keeps full-resolution host-memory use bounded without reducing input quality.
    # This host rejects some multi-GB cudaHostAlloc requests. Large inputs use
    # bounded pinned staging in to_device instead of pinning an entire queue.
    options = dict(num_workers=workers, pin_memory=False, collate_fn=collator)
    if workers:
        options.update(multiprocessing_context='spawn', prefetch_factor=1, persistent_workers=True)
    return DataLoader(dataset, **options, **kwargs)
