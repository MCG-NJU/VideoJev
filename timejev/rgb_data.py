"""Exact context budgeting and on-the-fly RGB inputs for the 192K recipe."""
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .decision_inputs import CHOICE, OPTION_END, task_of
from .engine import sha256
from .multiturn_inputs import continuation_text, episode_inputs
from .rgb_media import timeline, preprocess

CAPACITY = 192*1024
MIN_PIXELS = 320*320
MAX_PIXELS = 640*640
PROTOCOL = 'qwen35_rgb_dynamic_192k_v1'


@lru_cache(maxsize=2048)
def geometries(height, width, minimum=MIN_PIXELS):
    """Enumerate native resize outcomes; pixel limits are areas, not side lengths."""
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import smart_resize
    shapes = set()
    for cap in range(max(1024, minimum), MAX_PIXELS+1, 1024):
        h, w = smart_resize(2, height, width, min_pixels=2*minimum, max_pixels=2*cap)
        if minimum <= h*w <= MAX_PIXELS:
            shapes.add((h, w))
    if not shapes:
        raise ValueError(f'No native RGB size in pixel bounds: {height}x{width}')
    return sorted(shapes, key=lambda hw: (hw[0]*hw[1], hw), reverse=True)


def geometry(record, nonvisual_tokens, minimum=MIN_PIXELS, capacity=CAPACITY):
    groups = (record['frames']+1)//2
    for h, w in geometries(*record['original_hw'], minimum):
        visual = groups*(h//32)*(w//32)
        if visual+nonvisual_tokens <= capacity:
            return dict(processed_hw=[h, w], grid_thw=[groups, h//16, w//16],
                        tokens=visual, pixels_per_frame=h*w)
    return None


def text_lengths(processor, record, rows):
    """Exact nonvisual first-turn lengths and full continuation lengths.

    Timestamp option tails are tokenized once per video. Cuts are immediately
    before registered special tokens, so ordinary-text BPE merges are preserved.
    """
    from .timestamp_option_inputs import prompt, timestamp_texts
    from .token_metadata import _video_entry, exact_lengths
    groups = (record['frames']+1)//2
    skeleton = timeline(dict(record, grid_thw=[groups, 2, 2], tokens=groups, cache_id=PROTOCOL))
    timestamps = timestamp_texts(processor, skeleton)
    video = _video_entry(processor, skeleton)
    tails, heads, continued = {}, {}, []
    for row in rows:
        first = prompt(processor, row, timestamp_texts=timestamps)
        cont = continuation_text(processor, row, timestamps)
        if task_of(row) != CHOICE:
            first, marker, tail = first.partition(OPTION_END)
            cont, cont_marker, cont_tail = cont.partition(OPTION_END)
            if not marker or marker+tail != cont_marker+cont_tail:
                raise ValueError('First/continuation timestamp tails differ')
            tail = marker+tail
            if tail not in tails:
                tails[tail] = len(processor.tokenizer.encode(tail, add_special_tokens=False))
            extra = tails[tail]
        else:
            extra = 0
        heads[row['id']] = (first, extra)
        continued.append((cont, extra))
    first = exact_lengths(processor, rows, {record['video_key']: video},
                          lambda p, row: heads[row['id']][0])
    tokens = processor.tokenizer([text for text, _ in continued], add_special_tokens=False,
                                return_attention_mask=False, return_token_type_ids=False)['input_ids']
    first = [first[r['id']]['tokens']+heads[r['id']][1]-groups for r in rows]
    cont = [len(ids)+extra for ids, (_, extra) in zip(tokens, continued)]
    return first, cont


def plan_conversation(record, first, continuation, *, capacity=CAPACITY, minimum=MIN_PIXELS):
    """Compress first; split only if the entire conversation fails at the floor."""
    if not first or len(first) != len(continuation):
        raise ValueError('A conversation needs exact lengths for every question')
    whole = first[0]+sum(continuation[1:])
    shape = geometry(record, whole, minimum, capacity)
    if shape:
        return [dict(members=list(range(len(first))), geometry=shape, tokens=whole+shape['tokens'])], []
    pieces, dropped, start = [], [], 0
    while start < len(first):
        shape = geometry(record, first[start], minimum, capacity)
        if shape is None:
            dropped.append(start)
            start += 1
            continue
        end, nonvisual = start+1, first[start]
        while end < len(first) and geometry(record, nonvisual+continuation[end], minimum, capacity):
            nonvisual += continuation[end]
            end += 1
        shape = geometry(record, nonvisual, minimum, capacity)
        pieces.append(dict(members=list(range(start, end)), geometry=shape,
                           tokens=nonvisual+shape['tokens']))
        start = end
    return pieces, dropped


def text_item(processor, rows, record, episode):
    from .temporal import frame_labels
    record = timeline(dict(record, **episode['geometry']))
    inputs = episode_inputs(processor, record, rows)
    inputs['video_pixel_chunks'] = []  # Populated by RGBDataset; text-only evaluation builds later.
    if inputs['input_ids'].shape[1] != episode['tokens']:
        raise ValueError(f'Runtime RGB token length differs from plan: {episode["id"]}')
    times, labels = np.asarray(record['score_times']), []
    for row in rows:
        if task_of(row) == CHOICE:
            label = torch.zeros(len(row['options']))
            label[row['answer_index']] = 1
        else:
            label = torch.from_numpy(frame_labels(times, row['spans']))
        labels.append(label)
    return dict(rows=[dict(r, sampled_duration=record['sampled_duration']) for r in rows],
                inputs=inputs, times=times, labels=labels,
                audit=dict(input_source=record['source']['kind'], rgb_protocol=PROTOCOL,
                           rgb_source=record['source']['path'], feature_cache_id=PROTOCOL,
                           feature_path=f'{record["source"]["path"]}@{record["processed_hw"]}',
                           original_hw=record['original_hw'], processed_hw=record['processed_hw'],
                           pixels_per_frame=record['pixels_per_frame'], sampled_frames=record['frames'],
                           sample_fps=record['effective_fps'], requested_fps=record['fps'],
                           temporal_groups=len(times), visual_tokens=record['tokens'],
                           score_bounds=record['score_bounds'], score_timestamps=record['score_times'],
                           episode_id=episode['id'], max_sequence_length=CAPACITY))


class RGBDataset(Dataset):
    def __init__(self, config):
        from .data import read_jsonl
        self.config, self.processor = config, None
        self.rows = read_jsonl(config['train'])
        self.metadata_path = Path(config['rgb_metadata'])
        self.records = json.loads(self.metadata_path.read_text())['records']
        self.episode_spec = json.loads(Path(config['packed_train_lengths']).read_text())
        spec = self.episode_spec
        if (spec['manifest_sha256'] != sha256(config['train']) or
                spec['rgb_metadata_sha256'] != sha256(self.metadata_path) or
                spec['max_sequence_length'] != config['max_sequence_length']):
            raise ValueError('RGB dataset identity differs from its plan')
        self.episodes = spec['episodes']
        self.lengths = [e['tokens'] for e in self.episodes]
        if (sorted(i for e in self.episodes for i in e['query_indices']) != list(range(len(self.rows)))
                or min(self.lengths) <= 0 or max(self.lengths) > config['max_sequence_length']):
            raise ValueError('RGB episodes must cover every retained question exactly once')

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            index, *schedule = index
            return (dict(empty_pack=True) if index is None else self[index]) | {'pack_schedule': tuple(schedule)}
        if self.processor is None:
            from .decision_inputs import load_processor
            self.processor = load_processor(self.config['base'])
        episode = self.episodes[index]
        rows = [self.rows[i] for i in episode['query_indices']]
        record = self.records[episode['video_key']]
        item = text_item(self.processor, rows, record, episode)
        item['inputs']['video_pixel_chunks'] = preprocess(self.processor,
            dict(record, **episode['geometry']), self.config.get('rgb_chunk_frames', 32))
        return item


def single_video_collate(items):
    if len(items) != 1:
        raise ValueError('RGB evaluation prefetch owns one complete video per item')
    return items[0]


class RGBEvaluationVideos(Dataset):
    """Workers read/resize one video ahead; main process creates independent questions."""
    def __init__(self, config, spec, indices):
        self.config, self.spec, self.indices, self.processor = config, spec, indices, None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        if self.processor is None:
            from .decision_inputs import load_processor
            self.processor = load_processor(self.config['base'])
        index = self.indices[index]
        video = self.spec['videos'][index]
        chunks = preprocess(self.processor, dict(video['record'], **video['geometry']),
                            self.config.get('rgb_chunk_frames', 32))
        return dict(index=index, video_pixel_chunks=chunks)
