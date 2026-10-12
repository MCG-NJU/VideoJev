"""Question-normalized training batches with native video features shared per episode."""
import json
from pathlib import Path

import numpy as np
import torch

from .decision_data import DecisionDataset
from .decision_inputs import load_processor, CHOICE, task_of
from .engine import sha256
from .multiturn_inputs import LAYOUT, episode_inputs, episode_readouts
from .packing import PackedBatchSampler
from .temporal import frame_labels


def load_episode_spec(path, manifest, config, rows):
    spec = json.loads(Path(path).read_text())
    if (spec.get('training_layout') != LAYOUT or spec['manifest_sha256'] != sha256(manifest)
            or spec['max_sequence_length'] != config['max_sequence_length']
            or spec['seed'] != config['seed'] or spec['fps'] != config['train_fps']):
        raise ValueError('Multiturn plan differs from the training data/config')
    if spec.get('training_sources') != config.get('training_sources'):
        raise ValueError('Multiturn training cache routes differ from the metadata plan')
    from . import multiturn_inputs
    if spec['input_source_sha256'] != sha256(multiturn_inputs.__file__):
        raise ValueError('Multiturn input changed; regenerate the metadata plan')
    if any(sha256(p) != digest for p, digest in spec['source_bindings'].items()):
        raise ValueError('Multiturn feature/token metadata changed; regenerate the plan')
    episodes = spec['episodes']
    if len({e['video_key'] for e in episodes}) != len(episodes):
        raise ValueError('A video must not be split across episodes')
    flat = [i for e in episodes for i in e['query_indices']]
    if sorted(flat) != list(range(len(rows))):
        raise ValueError('Multiturn episodes duplicate or omit training queries')
    for episode in episodes:
        if (not episode['query_indices'] or not 0 < episode['tokens'] <= config['max_sequence_length']
                or any(rows[i]['video_key'] != episode['video_key'] for i in episode['query_indices'])):
            raise ValueError('Invalid episode membership or capacity')
    return spec


class MultiturnDataset(DecisionDataset):
    def __init__(self, manifest, base, *, episode_path, config, **kwargs):
        super().__init__(manifest, base, **kwargs)
        self.episode_spec = load_episode_spec(episode_path, manifest, config, self.rows)
        self.episodes = self.episode_spec['episodes']
        self.lengths = [e['tokens'] for e in self.episodes]

    def __len__(self):
        return len(self.episodes)

    def read_video_features(self, row):
        if self.feature_variant_root:
            from .adaptive_features import read_adaptive_features
            return read_adaptive_features(self.feature_root, self.feature_variant_root,
                row, self.fps, self.feature_spec['cache_id'], self.variant_spec, self.variant_plan)
        from .features import read_features
        return read_features(self.feature_root, row, self.fps, self.feature_spec['cache_id'])

    def __getitem__(self, index):
        if isinstance(index, tuple):
            index, *schedule = index
            return (dict(empty_pack=True) if index is None else self[index]) | {'pack_schedule': tuple(schedule)}
        if self.processor is None:
            self.processor = load_processor(self.base)
        episode = self.episodes[index]
        rows = [self.rows[i] for i in episode['query_indices']]
        features, record = self.read_video_features(rows[0])
        inputs = episode_inputs(self.processor, record, rows)
        if inputs['input_ids'].shape[1] != episode['tokens']:
            raise ValueError('Actual episode length differs from its metadata plan')
        inputs['visual_features'] = features
        times = np.asarray(record['score_times'], dtype=np.float64)
        labels = []
        for row in rows:
            if task_of(row) == CHOICE:
                label = torch.zeros(len(row['options']), dtype=torch.float32)
                label[row['answer_index']] = 1
            else:
                label = torch.from_numpy(frame_labels(times, row['spans']))
            labels.append(label)
        return dict(rows=[dict(r, sampled_duration=record['sampled_duration']) for r in rows],
            inputs=inputs, times=times, labels=labels,
            audit=dict(record['audit'], feature_cache_id=record['cache_id'], feature_path=record['path'],
                       training_layout=LAYOUT, episode_id=episode['id']))


class MultiturnMixtureDataset(MultiturnDataset):
    """Route each complete video conversation to its original native cache."""
    def __init__(self, config):
        from .data import read_jsonl
        self.base, self.fps = config['base'], config['train_fps']
        self.processor = None
        self.metadata_path = Path(config['train_media_shapes'])
        self.rows, self.sources_by_video = [], {}
        for source in config['training_sources']:
            child = DecisionDataset(source['manifest'], self.base, fps=self.fps,
                query_placement='last', metadata_path=source['media_shapes'],
                feature_root=source['feature_root'], feature_variant_root=source.get('feature_variant_root'),
                fast_feature_inputs=True)
            keys = {row['video_key'] for row in child.rows}
            if keys & self.sources_by_video.keys():
                raise ValueError('A video belongs to multiple training cache sources')
            self.sources_by_video.update({key: child for key in keys})
            self.rows.extend(child.rows)
        if self.rows != read_jsonl(config['train']):
            raise ValueError('Shared-prefix mixture must preserve all source rows in original order')
        self.episode_spec = load_episode_spec(config['packed_train_lengths'], config['train'], config, self.rows)
        self.episodes = self.episode_spec['episodes']
        self.lengths = [e['tokens'] for e in self.episodes]

    def read_video_features(self, row):
        return MultiturnDataset.read_video_features(self.sources_by_video[row['video_key']], row)


def training_dataset(config):
    if config.get('input_mode') == 'rgb_dynamic':
        from .rgb_data import RGBDataset
        return RGBDataset(config)
    if config.get('training_sources'):
        return MultiturnMixtureDataset(config)
    from .feature_sources import feature_sources
    root, variant = feature_sources(config, training=True)
    return MultiturnDataset(config['train'], config['base'], fps=config['train_fps'],
        query_placement='last', metadata_path=config['train_media_shapes'], feature_root=root,
        feature_variant_root=variant, fast_feature_inputs=True,
        episode_path=config['packed_train_lengths'], config=config)


def training_cache_specs(config):
    if config.get('training_sources'):
        from .timestamp_mixture_data import validate_training_caches
        return validate_training_caches(config)
    from .feature_sources import feature_sources
    from .features import validate_cache
    root, variant = feature_sources(config, training=True)
    specs = [validate_cache(root, [config['train']], config['train_fps'], replacement_root=variant)]
    if variant:
        specs.append(validate_cache(variant, [], config['train_fps'], require_complete=False))
    return specs


class MultiturnPackedBatchSampler(PackedBatchSampler):
    def __init__(self, *args, episodes, **kwargs):
        self.episodes = episodes
        super().__init__(*args, **kwargs)

    def plan(self):
        plan = super().plan()
        for entry in plan:
            if 'global_episode_ids' not in entry:
                entry['global_episode_ids'] = entry['global_ids']
                entry['global_ids'] = [q for e in entry['global_episode_ids']
                                       for q in self.episodes[e]['query_indices']]
        return plan


class MultiturnCollator:
    def __init__(self, tokenizer, max_length, *, pad_to_max_length=False, include_score_durations=False):
        self.include_score_durations = include_score_durations
        self.tokenizer, self.max_length, self.pad_to_max_length = tokenizer, max_length, pad_to_max_length

    def __call__(self, items):
        schedule = items[0].get('pack_schedule')
        if any(item.get('pack_schedule') != schedule for item in items):
            raise ValueError('Inconsistent packed update schedule')
        items = [x for x in items if not x.get('empty_pack')]
        lengths = [x['inputs']['input_ids'].shape[1] for x in items]
        turns = sum(len(x['rows']) for x in items)
        real = sum(lengths)
        size = max(64, (real + 63) // 64 * 64)
        if self.pad_to_max_length and items:
            size = max(size, self.max_length)
        if size > self.max_length:
            raise ValueError('Complete multi-turn pack exceeds capacity; truncation is forbidden')
        width = max((len(label) for x in items for label in x['labels']), default=1)
        ids = torch.full((1, size), self.tokenizer.pad_token_id, dtype=torch.long)
        types = torch.zeros_like(ids)
        pairs = torch.zeros((max(1, turns), width), dtype=torch.long)
        queries = torch.zeros(max(1, turns), dtype=torch.long)
        labels = torch.zeros((max(1, turns), width))
        durations = torch.zeros_like(labels) if self.include_score_durations else None
        mask = torch.zeros_like(labels, dtype=torch.bool)
        owners = torch.zeros(max(1, turns), dtype=torch.long)
        tasks = torch.zeros(max(1, turns), dtype=torch.long)
        rows, times, audits, offset, turn = [], [], [], 0, 0
        for owner, (item, length) in enumerate(zip(items, lengths)):
            if len(item['labels']) != len(item['rows']):
                raise ValueError('Every question requires its own label vector')
            source = item['inputs']['input_ids'][0]
            counts = [len(label) for label in item['labels']]
            p, q = episode_readouts(self.tokenizer, source, counts)
            ids[0, offset:offset+length] = source
            types[0, offset:offset+length] = item['inputs']['mm_token_type_ids'][0]
            for row, label, candidates, query in zip(item['rows'], item['labels'], p, q):
                task, count = task_of(row), len(label)
                expected = len(row['options']) if task == CHOICE else len(item['times'])
                if count != expected:
                    raise ValueError('Candidate count differs from the question task/labels')
                pairs[turn, :count], queries[turn] = candidates + offset, query + offset
                labels[turn, :count], mask[turn, :count] = label, True
                if durations is not None and task != CHOICE:
                    bounds = np.asarray(item['audit'].get('score_bounds'), dtype=np.float64)
                    if (bounds.shape != (count, 2) or not np.isfinite(bounds).all()
                            or np.any(bounds[:, 1] <= bounds[:, 0])
                            or np.any(bounds[1:, 0] != bounds[:-1, 1])):
                        raise ValueError('Invalid native time-cell bounds')
                    durations[turn, :count] = torch.from_numpy(bounds[:, 1] - bounds[:, 0])
                owners[turn], tasks[turn] = owner, task
                rows.append(row); times.append(item['times']); audits.append(item['audit'])
                turn += 1
            offset += length
        rgb = bool(items and 'video_pixel_chunks' in items[0]['inputs'])
        if any(('video_pixel_chunks' in x['inputs']) != rgb for x in items):
            raise ValueError('Cannot mix raw RGB and cached feature episodes')
        result = dict(input_ids=ids, mm_token_type_ids=types, attention_mask=None,
            video_grid_thw=torch.cat([x['inputs']['video_grid_thw'] for x in items]) if items else torch.empty((0, 3), dtype=torch.long),
            feature_cache_ids=[x['audit']['feature_cache_id'] for x in items],
            episode_lengths=lengths, packed_sample_lengths=lengths, turn_episode_indices=owners,
            pair_indices=pairs, query_indices=queries, score_mask=mask, labels=labels,
            decision_tasks=tasks, sample_count=turns, episode_count=len(items), real_tokens=real,
            pack_schedule=schedule, rows=rows, times=times, audits=audits)

        if rgb:
            result['video_pixel_chunks'] = [x['inputs']['video_pixel_chunks'] for x in items]
        else:
            result['visual_features'] = (torch.cat([x['inputs']['visual_features'] for x in items])
                                         if items else torch.empty((0, 2560), dtype=torch.bfloat16))

        if durations is not None:
            result['score_durations'] = durations
        return result
