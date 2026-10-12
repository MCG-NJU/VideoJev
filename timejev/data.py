"""Original video -> official processor; labels never enter prompt construction."""
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .temporal import frame_labels, pair_timeline
from .video import read_video
from .vision import load_processor, prepare_video


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


class GroundingDataset(Dataset):
    def __init__(self, manifest, base, limit=0, *, fps=2., archive_root=None, query_placement='both', metadata_path=None,
                 feature_root=None, require_complete=True, feature_variant_root=None,
                 fast_feature_inputs=False,visual_token_budget=None):
        self.rows = read_jsonl(manifest)
        if limit:
            self.rows = self.rows[:limit]
        self.base = base
        self.fps, self.archive_root = float(fps), archive_root
        self.processor = None
        self.feature_root = feature_root
        self.feature_variant_root=feature_variant_root
        self.fast_feature_inputs=fast_feature_inputs
        self.visual_token_budget=visual_token_budget
        if feature_root:
            from .features import validate_cache
            self.feature_spec = validate_cache(feature_root, [manifest], fps, require_complete=require_complete,
                                               replacement_root=feature_variant_root)
        if feature_variant_root:
            from .adaptive_features import load_adaptive_plan
            self.variant_spec,self.variant_plan=load_adaptive_plan(feature_variant_root,feature_root,[manifest],fps,
                                                                  require_complete=require_complete)
        self.query_placement = query_placement
        # Conservative bucketing estimate; actual tokens are never truncated.
        self.lengths = [int(np.ceil(r['duration'] * self.fps)) * 200 for r in self.rows]
        metadata_path = Path(metadata_path) if metadata_path else Path(manifest).parent / 'media_shapes.json'
        self.metadata_path = metadata_path
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
            self.lengths = [int(metadata.get(r['video_key'], {}).get('estimated_tokens', estimate * 2 / self.fps) * self.fps / 2)
                            for r, estimate in zip(self.rows, self.lengths)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if isinstance(index,tuple):
            idx,*schedule=index
            if idx is None:
                return dict(empty_pack=True,pack_schedule=tuple(schedule))
            return self[idx] | {'pack_schedule':tuple(schedule)}
        if self.processor is None:
            self.processor = load_processor(self.base)
        row = self.rows[index]
        if self.feature_root:
            from .features import read_features, cached_inputs
            if self.feature_variant_root:
                from .adaptive_features import read_adaptive_features
                features,record=read_adaptive_features(self.feature_root,self.feature_variant_root,row,self.fps,
                    self.feature_spec['cache_id'],self.variant_spec,self.variant_plan)
            else:
                features, record = read_features(self.feature_root, row, self.fps, self.feature_spec['cache_id'])
            if self.fast_feature_inputs:
                from .adaptive_features import fast_cached_inputs
                inputs=fast_cached_inputs(self.processor,record,row['query'],self.query_placement)
            else:
                inputs = cached_inputs(self.processor, record, row['query'], self.query_placement)
            inputs['visual_features'] = features
            times = np.asarray(record['score_times'], dtype=np.float64)
            audit = dict(record['audit'], query_placement=self.query_placement, input_source='visual_features',
                         feature_cache_id=record['cache_id'], feature_path=record['path'])
            if self.feature_variant_root:
                audit['visual_token_budget']=self.variant_spec['visual_token_budget']
            return dict(row=dict(row, sampled_duration=record['sampled_duration']), inputs=inputs, times=times,
                        labels=torch.from_numpy(frame_labels(times, row['spans'])), audit=audit)
        if self.archive_root:
            from .frame_archive import read_archive
            sample = read_archive(row, self.archive_root, self.fps)
        else:
            sample = read_video(row, self.fps)
        inputs, audit = prepare_video(self.processor, sample, row['query'], self.query_placement,
                                      visual_token_budget=self.visual_token_budget)
        times, bounds = pair_timeline(sample['times'], sample['sampled_duration'])
        audit.update(frame_timestamps=sample['times'].tolist(), score_timestamps=times.tolist(),
                     score_bounds=bounds.tolist(), label_rule='start <= pair_midpoint <= end')
        return dict(row=dict(row, sampled_duration=sample['sampled_duration']),
                    inputs=inputs, times=times,
                    labels=torch.from_numpy(frame_labels(times, row['spans'])),
                    audit=audit)


class Collator:
    def __init__(self, tokenizer):
        self.pad = tokenizer.pad_token_id
        self.vend = tokenizer.convert_tokens_to_ids('<|vision_end|>')
        self.video = tokenizer.convert_tokens_to_ids('<|video_pad|>')
        self.im_end = tokenizer.convert_tokens_to_ids('<|im_end|>')

    def __call__(self, items):
        lengths = [x['inputs']['input_ids'].shape[1] for x in items]
        length = (max(lengths) + 63) // 64 * 64
        batch = len(items)
        groups = max(int(x['inputs']['video_grid_thw'][0, 0]) for x in items)
        assert all(len(x['times']) == int(x['inputs']['video_grid_thw'][0, 0]) for x in items)
        ids = torch.full((batch, length), self.pad, dtype=torch.long)
        attention = torch.zeros_like(ids)
        modalities = torch.zeros_like(ids)
        pair_indices = torch.zeros((batch, groups), dtype=torch.long)
        query_indices = torch.empty(batch, dtype=torch.long)
        labels = torch.zeros((batch, groups), dtype=torch.float32)
        score_mask = torch.zeros((batch, groups), dtype=torch.bool)
        for b, (item, n) in enumerate(zip(items, lengths)):
            native = item['inputs']
            ids[b, :n] = native['input_ids'][0]
            attention[b, :n] = native['attention_mask'][0]
            modalities[b, :n] = native['mm_token_type_ids'][0]
            query_indices[b] = (ids[b, :n] == self.im_end).nonzero().flatten()[-1]
            # Only ends immediately following video tokens are temporal-group readouts.
            ends = ((ids[b, :n] == self.vend) & (torch.roll(ids[b, :n], 1) == self.video)).nonzero().flatten()
            assert len(ends) == int(native['video_grid_thw'][0, 0])
            assert query_indices[b] > ends[-1]
            pair_indices[b, :len(ends)] = ends
            count = len(item['times'])
            labels[b, :count] = item['labels']
            score_mask[b, :count] = True
        cached = ['visual_features' in x['inputs'] for x in items]
        assert all(cached) or not any(cached), 'Cannot mix cached/raw items in one batch'
        field = 'visual_features' if all(cached) else 'pixel_values_videos'
        values = [x['inputs'][field] for x in items]
        result = dict(input_ids=ids, attention_mask=attention, mm_token_type_ids=modalities,
                    video_grid_thw=torch.cat([x['inputs']['video_grid_thw'] for x in items]),
                    pair_indices=pair_indices, query_indices=query_indices,
                    labels=labels, score_mask=score_mask, rows=[x['row'] for x in items],
                    times=[x['times'] for x in items], audits=[x['audit'] for x in items])
        result[field] = values[0] if len(values) == 1 else torch.cat(values)
        if all(cached):
            result['feature_cache_ids'] = [x['audit']['feature_cache_id'] for x in items]
        return result


_STAGING_BYTES = 128 * 1024 * 1024
_STAGING = {}


def _copy_to_device(value, device):
    target = torch.device(device)
    if value.device.type != 'cpu' or target.type != 'cuda' or value.nbytes <= _STAGING_BYTES:
        return value.to(target, non_blocking=True)
    # Exact byte-preserving transfer through two bounded pinned buffers. The
    # input remains full precision/full size; no per-video multi-GB pin request.
    index = target.index if target.index is not None else torch.cuda.current_device()
    key = (index, value.dtype)
    count = _STAGING_BYTES // value.element_size()
    if key not in _STAGING:
        _STAGING[key] = ([torch.empty(count, dtype=value.dtype, pin_memory=True) for _ in range(2)], [None, None])
    buffers, events = _STAGING[key]
    flat = value.contiguous().view(-1)
    output = torch.empty(value.shape, dtype=value.dtype, device=target)
    result = output.view(-1)
    for chunk, start in enumerate(range(0, flat.numel(), count)):
        slot = chunk % 2
        if events[slot] is not None:
            events[slot].synchronize()
        size = min(count, flat.numel() - start)
        buffers[slot][:size].copy_(flat[start:start + size])
        result[start:start + size].copy_(buffers[slot][:size], non_blocking=True)
        events[slot] = torch.cuda.Event()
        events[slot].record(torch.cuda.current_stream(target))
    return output


def to_device(batch, device):
    return {k: _copy_to_device(v, device) if torch.is_tensor(v) else v for k, v in batch.items()}
