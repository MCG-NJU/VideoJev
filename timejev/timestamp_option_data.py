"""Dedicated timestamp-option input and collation; original datasets are unchanged."""
import numpy as np
import torch

from .decision_data import DecisionDataset as OriginalDataset, DecisionCollator as OriginalCollator
from .timestamp_option_inputs import (FORMULATION, GROUNDING_CANDIDATES, INPUT_PROTOCOL,
    CHOICE, load_processor, cached_inputs, task_of, readout_indices)
from .temporal import frame_labels


class TimestampOptionDataset(OriginalDataset):
    def __getitem__(self, index):
        if isinstance(index, tuple):
            idx, *schedule = index
            return (dict(empty_pack=True) if idx is None else self[idx]) | {'pack_schedule': tuple(schedule)}
        from .features import read_features
        if self.processor is None:
            self.processor = load_processor(self.base)
        row = self.rows[index]
        if self.feature_variant_root:
            from .adaptive_features import read_adaptive_features
            features, record = read_adaptive_features(self.feature_root, self.feature_variant_root, row, self.fps,
                self.feature_spec['cache_id'], self.variant_spec, self.variant_plan)
        else:
            features, record = read_features(self.feature_root, row, self.fps, self.feature_spec['cache_id'])
        inputs = cached_inputs(self.processor, record, row, fast=True)
        inputs['visual_features'] = features
        # Use exact existing score coordinates and labels, never the rounded
        # strings shown to Qwen as candidate text.
        times = np.asarray(record['score_times'], dtype=np.float64)
        if task_of(row) == CHOICE:
            labels = torch.zeros(len(row['options']), dtype=torch.float32)
            if row.get('answer_index') is not None:
                labels[row['answer_index']] = 1
        else:
            labels = torch.from_numpy(frame_labels(times, row.get('spans', [])))
        audit = dict(record['audit'], query_placement='last', input_source='visual_features',
            feature_cache_id=record['cache_id'], feature_path=record['path'], formulation=FORMULATION,
            decision_task=row.get('task', 'grounding'), grounding_candidates=GROUNDING_CANDIDATES,
            input_protocol=INPUT_PROTOCOL, timestamp_display='copied official <t:.1f seconds>',
            candidate_binding='native temporal group index; unrounded score_times and original bounds')
        if self.feature_variant_root:
            audit['visual_token_budget'] = self.variant_spec['visual_token_budget']
        return dict(row=dict(row, sampled_duration=record['sampled_duration']), inputs=inputs, times=times,
            labels=labels, audit=audit)


class TimestampOptionCollator(OriginalCollator):
    def __call__(self, items):
        schedule = items[0].get('pack_schedule')
        if any(item.get('pack_schedule') != schedule for item in items):
            raise ValueError('Inconsistent pack schedule')
        items = [item for item in items if not item.get('empty_pack', False)]
        ns = len(items)
        lengths = [item['inputs']['input_ids'].shape[1] for item in items]
        real_tokens = sum(lengths)
        size = real_tokens if self.packing else max(lengths, default=1)
        size = max(64, (size+63)//64*64)
        if self.pad_to_max_length and ns:
            size = max(size, self.max_length)
        if self.max_length and size > self.max_length:
            raise ValueError('Complete timestamp-option sample/pack exceeds capacity; truncation is forbidden')
        batch = 1 if self.packing else max(ns, 1)
        ids = torch.full((batch, size), self.tokenizer.pad_token_id, dtype=torch.long)
        types = torch.zeros_like(ids)
        attention = None if self.packing else torch.zeros_like(ids)
        counts = [len(item['labels']) for item in items]
        groups = max(counts, default=1)
        pairs = torch.zeros((max(ns, 1), groups), dtype=torch.long)
        queries = torch.zeros(max(ns, 1), dtype=torch.long)
        labels = torch.zeros((max(ns, 1), groups))
        durations = torch.zeros_like(labels) if self.include_score_durations else None
        mask = torch.zeros_like(labels, dtype=torch.bool)
        tasks = torch.zeros(max(ns, 1), dtype=torch.long)
        offset = 0
        for index, (item, length) in enumerate(zip(items, lengths)):
            source = item['inputs']['input_ids'][0]
            task = task_of(item['row'])
            candidate, query = readout_indices(self.tokenizer, source, task)
            if len(candidate) != counts[index]:
                raise ValueError('Suffix option readout count differs from target count')
            if task != CHOICE and (len(candidate) != len(item['times'])
                    or len(candidate) != int(item['inputs']['video_grid_thw'][0, 0])):
                raise ValueError('Timestamp options must map to every native temporal group')
            if durations is not None and task != CHOICE:
                bounds = np.asarray(item['audit'].get('score_bounds'), dtype=np.float64)
                if (bounds.shape != (counts[index], 2) or not np.isfinite(bounds).all()
                        or np.any(bounds[:, 1] <= bounds[:, 0])
                        or np.any(bounds[1:, 0] != bounds[:-1, 1])):
                    raise ValueError('Soft IoU requires original contiguous positive-length score bounds')
                durations[index, :counts[index]] = torch.from_numpy(bounds[:, 1]-bounds[:, 0])
            row, start = (0, offset) if self.packing else (index, 0)
            ids[row, start:start+length] = source
            types[row, start:start+length] = item['inputs']['mm_token_type_ids'][0]
            if attention is not None:
                attention[row, :length] = 1
            pairs[index, :len(candidate)] = candidate+start
            queries[index] = query+start
            labels[index, :counts[index]], mask[index, :counts[index]], tasks[index] = item['labels'], True, task
            offset += length
        result = dict(input_ids=ids, attention_mask=attention, mm_token_type_ids=types,
            video_grid_thw=torch.cat([item['inputs']['video_grid_thw'] for item in items]) if ns else torch.empty((0, 3), dtype=torch.long),
            visual_features=torch.cat([item['inputs']['visual_features'] for item in items]) if ns else torch.empty((0, 2560), dtype=torch.bfloat16),
            feature_cache_ids=[item['audit']['feature_cache_id'] for item in items],
            pair_indices=pairs, query_indices=queries, labels=labels, score_mask=mask, decision_tasks=tasks,
            rows=[item['row'] for item in items], times=[item['times'] for item in items], audits=[item['audit'] for item in items])
        if durations is not None:
            result['score_durations'] = durations
        if self.packing:
            from xtuner.v1.datasets.collator import split_to_chunks
            all_lengths = lengths+split_to_chunks(size-real_tokens, 256)
            cu = torch.tensor([0, *torch.tensor(all_lengths).cumsum(0).tolist()], dtype=torch.int32)
            result.update(packed_sample_lengths=lengths, cu_seq_lens_q=cu, cu_seq_lens_k=cu,
                max_length_q=max(all_lengths), max_length_k=max(all_lengths),
                seq_idx=torch.repeat_interleave(torch.arange(len(all_lengths), dtype=torch.int32),
                    torch.tensor(all_lengths))[None, :],
                sample_count=ns, real_tokens=real_tokens, pack_schedule=schedule)
        return result


class PackedTimestampOptionCollator(TimestampOptionCollator):
    def __init__(self, tokenizer, max_length, *, pad_to_max_length=False, include_score_durations=False):
        super().__init__(tokenizer, max_length, packing=True, pad_to_max_length=pad_to_max_length,
            include_score_durations=include_score_durations)


def load_lengths(path, manifest, config, rows):
    import json
    from pathlib import Path
    from . import timestamp_option_inputs
    from .engine import sha256
    from .packing import load_lengths as native_load_lengths
    from .timestamp_option_inputs import token_metadata
    spec = json.loads(Path(path).read_text())
    if config.get('grounding_candidates') != GROUNDING_CANDIDATES:
        raise ValueError('Timestamp-option lengths require their explicit candidate setting')
    if spec.get('decision_metadata') != token_metadata(load_processor(config['base']).tokenizer):
        raise ValueError('Lengths were not built with the timestamp-option prompt/tokenizer')
    expected = dict(formulation=FORMULATION, source_sha256=sha256(timestamp_option_inputs.__file__))
    if spec.get('length_metadata', {}).get('identity', {}).get('prompt_version') != expected:
        raise ValueError('Timestamp-option input source changed; regenerate metadata lengths')
    training = Path(manifest).resolve() == Path(config['train']).resolve()
    if spec.get('fps') != config['train_fps' if training else 'eval_fps']:
        raise ValueError('Timestamp-option lengths belong to another sampling rate')
    return native_load_lengths(path, manifest, config, rows)


DecisionDataset = TimestampOptionDataset
DecisionCollator = TimestampOptionCollator
PackedDecisionCollator = PackedTimestampOptionCollator
