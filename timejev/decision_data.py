"""Grounding and choice samples share native features and isolated packing."""
import numpy as np
import torch
from .data import GroundingDataset as NativeDataset
from .decision_inputs import load_processor, cached_inputs, task_of, validate_row, readout_indices, CHOICE, FORMULATION
from .temporal import frame_labels


class DecisionDataset(NativeDataset):
    def __init__(self, *args, require_labels=True, **kwargs):
        super().__init__(*args, **kwargs)
        if self.query_placement != 'last':
            raise ValueError('Decision formulation uses exactly one trailing question')
        if not self.feature_root:
            raise ValueError('DecisionDataset requires native precomputed visual features')
        self.require_labels = require_labels
        for row in self.rows:
            validate_row(row, require_labels=require_labels)

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
        times = np.asarray(record['score_times'], dtype=np.float64)
        if task_of(row) == CHOICE:
            labels = torch.zeros(len(row['options']), dtype=torch.float32)
            if row.get('answer_index') is not None:
                labels[row['answer_index']] = 1
        else:
            labels = torch.from_numpy(frame_labels(times, row.get('spans', [])))
        audit = dict(record['audit'], query_placement='last', input_source='visual_features',
            feature_cache_id=record['cache_id'], feature_path=record['path'], formulation=FORMULATION,
            decision_task=row.get('task', 'grounding'))
        if self.feature_variant_root:
            audit['visual_token_budget'] = self.variant_spec['visual_token_budget']
        return dict(row=dict(row, sampled_duration=record['sampled_duration']), inputs=inputs, times=times,
                    labels=labels, audit=audit)


class DecisionCollator:
    def __init__(self, tokenizer, max_length=None, *, packing=False, pad_to_max_length=False,
                 include_score_durations=False):
        self.tokenizer, self.max_length = tokenizer, max_length
        self.packing, self.pad_to_max_length = packing, pad_to_max_length
        self.include_score_durations = include_score_durations

    def __call__(self, items):
        schedule = items[0].get('pack_schedule')
        if any(x.get('pack_schedule') != schedule for x in items):
            raise ValueError('Inconsistent pack schedule')
        items = [x for x in items if not x.get('empty_pack', False)]
        ns = len(items)
        lengths = [x['inputs']['input_ids'].shape[1] for x in items]
        real_tokens = sum(lengths)
        size = real_tokens if self.packing else max(lengths, default=1)
        size = max(64, (size + 63) // 64 * 64)
        if self.pad_to_max_length and ns:
            size = max(size, self.max_length)
        if self.max_length and size > self.max_length:
            raise ValueError('Complete samples exceed sequence capacity; truncation is forbidden')
        batch = 1 if self.packing else max(ns, 1)
        ids = torch.full((batch, size), self.tokenizer.pad_token_id, dtype=torch.long)
        types = torch.zeros_like(ids)
        attention = None if self.packing else torch.zeros_like(ids)
        counts = [len(x['labels']) for x in items]
        groups = max(counts, default=1)
        pairs = torch.zeros((max(ns, 1), groups), dtype=torch.long)
        queries = torch.zeros(max(ns, 1), dtype=torch.long)
        labels = torch.zeros((max(ns, 1), groups))
        durations = torch.zeros_like(labels) if self.include_score_durations else None
        mask = torch.zeros_like(labels, dtype=torch.bool)
        tasks = torch.zeros(max(ns, 1), dtype=torch.long)
        offset = 0
        for i, (x, n) in enumerate(zip(items, lengths)):
            src = x['inputs']['input_ids'][0]
            task = task_of(x['row'])
            candidate, query = readout_indices(self.tokenizer, src, task)
            if len(candidate) != counts[i]:
                raise ValueError('Candidate readout count differs from target count')
            if task != CHOICE and len(candidate) != len(x['times']):
                raise ValueError('Temporal readouts differ from native pair timestamps')
            if durations is not None and task != CHOICE:
                bounds = np.asarray(x['audit'].get('score_bounds'), dtype=np.float64)
                if (bounds.shape != (counts[i], 2) or not np.isfinite(bounds).all() or
                        np.any(bounds[:, 1] <= bounds[:, 0]) or
                        np.any(bounds[1:, 0] != bounds[:-1, 1])):
                    raise ValueError('Soft IoU requires finite, contiguous, positive-length score_bounds')
                durations[i, :counts[i]] = torch.from_numpy(bounds[:, 1] - bounds[:, 0])
            b, start = (0, offset) if self.packing else (i, 0)
            ids[b, start:start+n] = src
            types[b, start:start+n] = x['inputs']['mm_token_type_ids'][0]
            if attention is not None:
                attention[b, :n] = 1
            pairs[i, :len(candidate)] = candidate + start
            queries[i] = query + start
            labels[i, :counts[i]], mask[i, :counts[i]], tasks[i] = x['labels'], True, task
            offset += n
        result = dict(input_ids=ids, attention_mask=attention, mm_token_type_ids=types,
            video_grid_thw=torch.cat([x['inputs']['video_grid_thw'] for x in items]) if ns else torch.empty((0,3),dtype=torch.long),
            visual_features=torch.cat([x['inputs']['visual_features'] for x in items]) if ns else torch.empty((0,2560),dtype=torch.bfloat16),
            feature_cache_ids=[x['audit']['feature_cache_id'] for x in items],
            pair_indices=pairs, query_indices=queries, labels=labels, score_mask=mask, decision_tasks=tasks,
            rows=[x['row'] for x in items], times=[x['times'] for x in items], audits=[x['audit'] for x in items])
        if durations is not None:
            result['score_durations'] = durations
        if self.packing:
            from xtuner.v1.datasets.collator import split_to_chunks
            all_lengths = lengths + split_to_chunks(size-real_tokens, 256)
            cu = torch.tensor([0, *torch.tensor(all_lengths).cumsum(0).tolist()], dtype=torch.int32)
            result.update(packed_sample_lengths=lengths, cu_seq_lens_q=cu, cu_seq_lens_k=cu,
                max_length_q=max(all_lengths), max_length_k=max(all_lengths),
                seq_idx=torch.repeat_interleave(torch.arange(len(all_lengths),dtype=torch.int32),torch.tensor(all_lengths))[None,:],
                sample_count=ns, real_tokens=real_tokens, pack_schedule=schedule)
        return result


class PackedDecisionCollator(DecisionCollator):
    def __init__(self, tokenizer, max_length, *, pad_to_max_length=False, include_score_durations=False):
        super().__init__(tokenizer, max_length, packing=True, pad_to_max_length=pad_to_max_length,
                         include_score_durations=include_score_durations)


def load_lengths(path, manifest, config, rows):
    import json
    from pathlib import Path
    from . import decision_inputs
    from .engine import sha256
    from .packing import load_lengths as native_load_lengths
    from .decision_inputs import token_metadata
    spec = json.loads(Path(path).read_text())
    if spec.get('decision_metadata') != token_metadata(load_processor(config['base']).tokenizer):
        raise ValueError('Lengths were not built with the current decision prompt/tokenizer')
    expected_prompt = dict(formulation=FORMULATION, source_sha256=sha256(decision_inputs.__file__))
    if spec.get('length_metadata', {}).get('identity', {}).get('prompt_version') != expected_prompt:
        raise ValueError('Decision prompt source changed; regenerate length metadata before packing')
    training = Path(manifest).resolve() == Path(config['train']).resolve()
    if spec.get('fps') != config['train_fps' if training else 'eval_fps']:
        raise ValueError('Decision lengths belong to another sampling rate')
    return native_load_lengths(path, manifest, config, rows)
