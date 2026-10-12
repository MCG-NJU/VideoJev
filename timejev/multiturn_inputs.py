"""Causal decision turns sharing one native video prefix, without answer history."""
import hashlib
import json

import numpy as np
import torch

from .decision_inputs import CHOICE, DECIDE, OPTION_END, task_of, validate_row
from .timestamp_option_inputs import cached_inputs, suffix, timestamp_texts

LAYOUT = 'shared_video_multiturn_v1'
GROUPINGS = ('video', 'record')


def grouping_mode(config):
    mode = config.get('multiturn_group_by', 'video')
    if mode not in GROUPINGS:
        raise ValueError(f'multiturn_group_by must be one of {GROUPINGS}')
    return mode


def group_memberships(rows, group_by='video'):
    """Keep manifest order, and keep every original record intact in record mode."""
    if group_by not in GROUPINGS or not rows:
        raise ValueError('Expected nonempty rows and video/record grouping')
    groups, videos, records, ids = {}, {}, {}, set()
    for index, row in enumerate(rows):
        if row['id'] in ids:
            raise ValueError(f'Duplicate question ID: {row["id"]}')
        ids.add(row['id'])
        video = row['video_key']
        identity = (row['video_path'], float(row['duration']))
        if videos.setdefault(video, identity) != identity:
            raise ValueError('A video key identifies different media/durations')
        record = None
        if 'source_id' in row:
            if not isinstance(row['source_id'], str) or not row['source_id']:
                raise ValueError('A source record needs a nonempty source_id')
            record = (row.get('source_file', row.get('source', '')), row['source_id'])
            previous_video, turns = records.setdefault(record, (video, []))
            if previous_video != video:
                raise ValueError('An original JSONL record contains different videos')
            turn = row.get('turn_index')
            if type(turn) is not int or turn != len(turns):
                raise ValueError('Original record turns must be complete and in source order')
            turns.append(index)
        elif group_by == 'record':
            raise ValueError('Record grouping requires source_id and turn_index')
        key = video if group_by == 'video' else record
        groups.setdefault(key, []).append(index)
    episodes = []
    for key, indices in groups.items():
        identity = (str(key) if group_by == 'video' else
                    hashlib.sha256(json.dumps(key, ensure_ascii=False).encode()).hexdigest())
        episodes.append(dict(id=f'{group_by}/{identity}', video_key=rows[indices[0]]['video_key'],
                             query_indices=indices))
    return episodes


def group_episodes(rows, first_lengths, continuation_lengths, capacity, group_by='video'):
    if len(rows) != len(first_lengths) or len(rows) != len(continuation_lengths):
        raise ValueError('Every question needs both first-turn and continuation lengths')
    episodes = group_memberships(rows, group_by)
    for episode in episodes:
        indices = episode['query_indices']
        length = first_lengths[indices[0]] + sum(continuation_lengths[i] for i in indices[1:])
        if length <= 0 or (capacity is not None and length > capacity):
            raise ValueError(f'Complete episode {episode["id"]} needs {length} tokens; '
                             f'capacity is {capacity}. No turns were truncated or split.')
        episode['tokens'] = int(length)
    return episodes


def continuation_text(processor, row, timestamps=None):
    validate_row(row, require_labels=True)
    return processor.apply_chat_template([dict(role='user', content=[
        dict(type='text', text=suffix(row, timestamp_texts=timestamps))])],
        tokenize=False, add_generation_prompt=False)


def episode_inputs(processor, record, rows):
    if not rows:
        raise ValueError('An episode needs at least one question')
    for row in rows:
        validate_row(row, require_labels=True)
    first = cached_inputs(processor, record, rows[0])
    timestamps = (timestamp_texts(processor, record)
                  if any(task_of(row) != CHOICE for row in rows[1:]) else None)
    parts, types = [first['input_ids']], [first['mm_token_type_ids']]
    for row in rows[1:]:
        ids = processor.tokenizer.encode(continuation_text(processor, row, timestamps),
                                         add_special_tokens=False)
        value = torch.tensor([ids], dtype=torch.long)
        parts.append(value)
        types.append(torch.zeros_like(value))
    ids = torch.cat(parts, dim=1)
    return dict(input_ids=ids, mm_token_type_ids=torch.cat(types, dim=1),
                attention_mask=torch.ones_like(ids), video_grid_thw=first['video_grid_thw'])


def episode_readouts(tokenizer, ids, candidate_counts):
    """Return ragged per-turn candidates; candidate counts may differ by task/turn."""
    decisions = (ids == tokenizer.convert_tokens_to_ids(DECIDE)).nonzero().flatten()
    options = (ids == tokenizer.convert_tokens_to_ids(OPTION_END)).nonzero().flatten()
    if (not candidate_counts or min(candidate_counts) < 1 or
            len(decisions) != len(candidate_counts) or len(options) != sum(candidate_counts)):
        raise ValueError('Each turn needs its own DECIDE and complete candidate list')
    groups, offset, previous = [], 0, -1
    for count, decide in zip(candidate_counts, decisions):
        candidates = options[offset:offset+count]
        if not ((candidates > previous) & (candidates < decide)).all():
            raise ValueError('Candidate readouts cross turn boundaries')
        groups.append(candidates)
        offset += count
        previous = decide
    return groups, decisions


def record_from_identity(identity):
    record = dict(identity)
    frames, fps = record['frames'], record.get('effective_fps', record['fps'])
    first = np.arange(0, frames, 2, dtype=np.float64)
    record['score_times'] = ((first + np.minimum(first + 1, frames - 1)) / (2 * fps)).tolist()
    return record
