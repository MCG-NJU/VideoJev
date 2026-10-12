"""Lossless cache of native Qwen3.5 merger outputs, with explicit provenance."""
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
from transformers.video_utils import VideoMetadata

from .engine import save_json, sha256
from .frame_archive import fingerprint
from .temporal import pair_timeline
from .vision import VISUAL_PROTOCOL, prompt

FEATURE_PROTOCOL = 'qwen35_native_merger_bf16_v1'


def state_digest(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().float().contiguous()
        h.update(name.encode())
        h.update(str(tuple(value.shape)).encode())
        h.update(memoryview(value.numpy()))
    return h.hexdigest()


def pretrained_vision(base):
    config = AutoConfig.from_pretrained(base).vision_config
    config._attn_implementation = 'flash_attention_2'
    config.dtype = torch.float32
    model = Qwen3_5VisionModel(config)
    index = json.loads((Path(base) / 'model.safetensors.index.json').read_text())['weight_map']
    state = {}
    for filename in sorted({v for k, v in index.items() if k.startswith('model.visual.')}):
        with safe_open(str(Path(base) / filename), framework='pt') as source:
            for key in source.keys():
                if key.startswith('model.visual.'):
                    state[key.removeprefix('model.visual.')] = source.get_tensor(key).float()
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False)


def cache_key(row, fps):
    spec = [str(Path(row['video_path']).resolve()), float(row['duration']), float(fps)]
    return hashlib.sha256(json.dumps(spec).encode()).hexdigest()


def cache_paths(root, row, fps):
    from .feature_catalog import resolve
    root, _ = resolve(root, row, fps)
    key = cache_key(row, fps)
    path = Path(root) / 'tensors' / key[:2] / key
    return path.with_suffix('.safetensors'), path.with_suffix('.json')


def cached_inputs(processor, record, query, placement='both'):
    grid = torch.tensor([record['grid_thw']], dtype=torch.long)
    metadata = VideoMetadata(total_num_frames=record['frames'], fps=record.get('effective_fps',record['fps']),
                             frames_indices=list(range(record['frames'])))
    replacement = processor.replace_video_token(
        dict(video_grid_thw=grid, video_metadata=[metadata]), video_idx=0)
    texts, _ = processor.get_text_with_replacements([prompt(processor, query, placement)], [], [replacement], [])
    # The official text path adds modality IDs; no visual preprocessor is invoked.
    inputs = processor(text=texts, return_tensors='pt')
    assert int((inputs['input_ids'] == processor.video_token_id).sum()) == record['tokens']
    return {k: inputs[k] for k in ['input_ids', 'attention_mask', 'mm_token_type_ids']} | {'video_grid_thw': grid}


def write_features(root, row, fps, sample, inputs, audit, features, cache_id):
    if features.dtype != torch.bfloat16:
        raise ValueError(f'Expected native BF16 merger output, got {features.dtype}')
    features = features.detach().cpu().contiguous()
    grid = inputs['video_grid_thw'].cpu()
    assert features.shape == (int(grid.prod()) // 4, 2560)
    assert torch.isfinite(features).all()
    path, metadata_path = cache_paths(root, row, fps)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
    centers, bounds = pair_timeline(sample['times'], sample['sampled_duration'])
    audit = dict(audit, frame_timestamps=sample['times'].tolist(), score_timestamps=centers.tolist(),
                 score_bounds=bounds.tolist(), label_rule='start <= pair_midpoint <= end')
    record = dict(protocol=FEATURE_PROTOCOL, visual_protocol=VISUAL_PROTOCOL, cache_id=cache_id,
                  video_key=row['video_key'], source=fingerprint(row['video_path']),
                  requested_duration=float(row['duration']), fps=float(fps), frames=len(sample['times']),
                  grid_thw=grid[0].tolist(), tokens=features.shape[0], dimension=2560, dtype='bfloat16',
                  times=sample['times'].tolist(), score_times=centers.tolist(), bounds=bounds.tolist(),
                  sampled_duration=sample['sampled_duration'], audit=audit, path=str(path.resolve()))
    if 'max_frames' in sample:
        record.update(effective_fps=sample['fps'],requested_fps=sample['requested_fps'],
                      max_frames=sample['max_frames'],sampling_mode=sample['sampling_mode'],
                      sampling_duration=sample['sampling_duration'])
    try:
        save_file({'features': features, 'video_grid_thw': grid}, str(temporary),
                  metadata={'protocol': FEATURE_PROTOCOL, 'cache_id': cache_id})
        restored = load_file(str(temporary))
        assert torch.equal(restored['features'], features)
        assert torch.equal(restored['video_grid_thw'], grid)
        record.update(bytes=temporary.stat().st_size, sha256=sha256(temporary), verified=True)
        os.replace(temporary, path)
        save_json(record, metadata_path)
    finally:
        temporary.unlink(missing_ok=True)
    return record


def read_features(root, row, fps, cache_id, *, verify=False):
    from .feature_catalog import resolve
    root, cache_id = resolve(root, row, fps, cache_id)
    path, metadata_path = cache_paths(root, row, fps)
    record = json.loads(metadata_path.read_text())
    if (record['protocol'] != FEATURE_PROTOCOL or record['cache_id'] != cache_id
            or record['fps'] != float(fps) or record['requested_duration'] != float(row['duration'])
            or record['source'] != fingerprint(row['video_path']) or not record['verified']):
        raise ValueError('Feature cache identity/source/config mismatch')
    if path.stat().st_size != record['bytes'] or (verify and sha256(path) != record['sha256']):
        raise ValueError('Feature cache file changed or truncated')
    tensors = load_file(str(path))
    value = tensors['features']
    assert value.dtype == torch.bfloat16 and tuple(value.shape) == (record['tokens'], 2560)
    assert tensors['video_grid_thw'].tolist() == [record['grid_thw']]
    return value, record


def validate_cache(root, manifests, fps, *, require_complete=True, replacement_root=None):
    spec = json.loads((Path(root) / 'provenance.json').read_text())
    from .feature_catalog import LAYOUT, validate
    if spec.get('cache_layout') == LAYOUT:
        return validate(root, manifests, fps, replacement_root=replacement_root)
    assert spec['protocol'] == FEATURE_PROTOCOL
    if require_complete:
        index = json.loads((Path(root) / 'index.json').read_text())
        assert index['complete'] and index['cache_id'] == spec['cache_id']
        keys = set(index['entries'])
        retirement_path=Path(root)/'retired_payloads.json'
        retired=json.loads(retirement_path.read_text()) if retirement_path.exists() else None
        replacement=None
        if retired:
            assert retired['base_cache_id']==spec['cache_id']
            if replacement_root:
                replacement=json.loads((Path(replacement_root)/'index.json').read_text())
                assert replacement['complete'] and replacement['cache_id']==retired['replacement_cache_id']
        for manifest in manifests:
            # Queries sharing a source and duration share one cache entry. Keep
            # canonical path checks, but avoid repeated serial remote lstat calls.
            rows = {}
            for line in Path(manifest).read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    rows[(row['video_path'], float(row['duration']))] = row
            with ThreadPoolExecutor(max_workers=8) as pool:
                for key in pool.map(lambda row: cache_key(row, fps), rows.values()):
                    assert key in keys, (manifest, fps, key)
                    if retired and key in retired['entries']:
                        if replacement is None:
                            raise ValueError('Original oversized features were retired; use the matching adaptive cache or raw video for historical resolution')
                        record=replacement['entries'][key]
                        assert record['sha256']==retired['entries'][key]['replacement_sha256']
                        assert Path(record['path']).stat().st_size==record['bytes']
    return spec
