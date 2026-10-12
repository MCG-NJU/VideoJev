"""Consume the completed 4B QA ViT caches without opening JPEGs or videos."""
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch.utils.data import Dataset

from .context_eval import CONTEXTS, COUNTS, VIDEO_COUNTS, digest_json, validate_spec
from .data import read_jsonl
from .engine import sha256
from .features import state_digest

CACHE_PROTOCOL = 'qwen35_4b_qa_context80_merger_bf16_v1'


def read(path):
    return json.loads(Path(path).read_text())


def validate_checkpoint_vision(checkpoint, root):
    model = read(Path(root)/'model.json')
    state = {k.removeprefix('backbone.visual.'): v for k, v in checkpoint['model'].items()
             if k.startswith('backbone.visual.')}
    if (not checkpoint['config']['freeze_visual'] or not state
            or state_digest(state) != model['vision_state_sha256']):
        raise ValueError('Checkpoint ViT/merger weights differ from the offline QA cache')


def validate_feature_cache(root, plans, base):
    """Check metadata/coverage only; tensor integrity is checked when consumed."""
    root, plans = Path(root), Path(plans)
    model, complete = read(root/'model.json'), read(root/'completion.json')
    if (not complete.get('complete') or complete['protocol'] != CACHE_PROTOCOL
            or complete['model_sha256'] != model['weights_sha256']
            or model['base'] != base or model['dimension'] != 2560
            or model['output_dtype'] != 'bfloat16'):
        raise ValueError('Incomplete or incompatible offline QA cache')
    specs = {}
    for context in CONTEXTS:
        plan = read(plans/f'context_{context}.json')
        for name, count in COUNTS.items():
            spec = plan['benchmarks'][name]
            rows = read_jsonl(spec['manifest'])
            validate_spec(spec, rows, context)
            cache_plan_path = root/f'plans/{context}/{name}.json'
            cache_plan = read(cache_plan_path)
            index = read(root/f'index/{context}/{name}.json')
            if (len(rows) != count or len(spec['videos']) != VIDEO_COUNTS[name]
                    or sha256(spec['manifest']) != spec['manifest_sha256']
                    or cache_plan['manifest_sha256'] != spec['manifest_sha256']
                    or cache_plan['model_sha256'] != model['weights_sha256']
                    or not index.get('complete') or index['context'] != context
                    or index['benchmark'] != name or index['protocol'] != CACHE_PROTOCOL
                    or index['model_sha256'] != model['weights_sha256']
                    or index['plan_sha256'] != sha256(cache_plan_path)):
                raise ValueError(f'Cache plan/index identity differs: {context}/{name}')
            expected = [dict(video_key=v['video_key'],
                feature_key=digest_json(dict(record=v['record'], geometry=v['geometry'])),
                geometry=v['geometry'], questions=v['questions']) for v in spec['videos']]
            if (cache_plan['videos'] != expected
                    or set(index['entries']) != {v['video_key'] for v in expected}):
                raise ValueError(f'Cache videos/questions differ: {context}/{name}')
            for video in expected:
                entry = index['entries'][video['video_key']]
                path = Path(entry['path'])
                if (entry['key'] != video['feature_key'] or entry['dimension'] != 2560
                        or entry['tokens'] != video['geometry']['tokens']
                        or entry['grid_thw'] != video['geometry']['grid_thw']
                        or not path.is_file() or path.stat().st_size != entry['bytes']):
                    raise ValueError(f'Missing/changed cached feature: {path}')
            specs[(context, name)] = index
    return specs


class CachedEvaluationVideos(Dataset):
    def __init__(self, index, spec, indices):
        self.index, self.spec, self.indices = index, spec, indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        video = self.spec['videos'][index]
        entry = self.index['entries'][video['video_key']]
        path = Path(entry['path'])
        # Verify exactly the bytes loaded; do not re-encode or silently fall back.
        if path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
            raise ValueError(f'Cached feature checksum changed: {path}')
        record = read(path.with_suffix('.json'))
        if (record['model_sha256'] != self.index['model_sha256']
                or record['key'] != entry['key'] or record['sha256'] != entry['sha256']
                or not record['verified']):
            raise ValueError(f'Cached feature provenance differs: {path}')
        tensors = load_file(str(path))
        features = tensors['features']
        if (features.dtype != torch.bfloat16 or features.shape != (entry['tokens'], 2560)
                or tensors['video_grid_thw'].tolist() != [entry['grid_thw']]):
            raise ValueError(f'Cached feature shape/dtype differs: {path}')
        return dict(index=index, visual_features=features)


def single_video_collate(items):
    if len(items) != 1:
        raise ValueError('Evaluation owns one complete video per batch')
    return items[0]
