import os
"""Coverage, native templates, ragged choices and per-question update semantics."""
import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from test_helpers import record_for
from timejev.decision_model import typed_outputs
from timejev.multiturn_data import MultiturnCollator, MultiturnDataset, MultiturnPackedBatchSampler, load_episode_spec
from timejev.multiturn_inputs import (LAYOUT, continuation_text, episode_inputs,
                                    episode_readouts, group_episodes, group_memberships)
from timejev.timestamp_option_data import PackedTimestampOptionCollator
from timejev.timestamp_option_inputs import load_processor, suffix, timestamp_texts, _replacement
from timejev.temporal import frame_labels


@pytest.fixture(scope='module')
def processor():
    model = os.environ.get('VIDEOJEV_TEST_MODEL', 'models/Qwen3.5-4B')
    if not Path(model).is_dir():
        pytest.skip('Set VIDEOJEV_TEST_MODEL to a local Qwen3.5-4B model')
    return load_processor(model)


def make_episode(processor, counts=(4, 2, 5), frames=4, identity='video', hidden=128):
    record = record_for(frames)
    rows, labels = [], []
    for i, count in enumerate(counts):
        row = dict(id=f'{identity}-{i}', source='test', source_id=identity, turn_index=i,
            video_key=identity, video_path=f'/{identity}.mp4', duration=record['sampled_duration'],
            sampled_duration=record['sampled_duration'], query=f'What action happens in scene {i}?')
        if count is None:
            row.update(task='grounding', spans=[[0., .3]])
            labels.append(torch.from_numpy(frame_labels(np.asarray(record['score_times']), row['spans'])))
        else:
            row.update(task='choice', options=[f'{chr(65+j)}. Action number {j}.' for j in range(count)],
                       answer_index=i % count)
            label = torch.zeros(count); label[row['answer_index']] = 1
            labels.append(label)
        rows.append(row)
    inputs = episode_inputs(processor, record, rows)
    inputs['visual_features'] = torch.randn(record['tokens'], hidden, dtype=torch.bfloat16)
    return dict(rows=rows, inputs=inputs, times=np.asarray(record['score_times']), labels=labels,
                audit=dict(feature_cache_id='test', feature_path=f'{identity}.features',
                           score_bounds=record['audit']['score_bounds']))


def grouping_rows():
    result = []
    for video, record, turns in [('v', 'r1', 2), ('other', 'r2', 1), ('v', 'r3', 3)]:
        for i in range(turns):
            result.append(dict(id=f'{record}/{i}', source='file1', source_id=record, turn_index=i,
                               video_key=video, video_path=f'/{video}', duration=10))
    return result


def test_both_groupings_keep_exact_source_order_and_coverage():
    rows = grouping_rows()
    by_video = group_memberships(rows, 'video')
    by_record = group_memberships(rows, 'record')
    assert [e['query_indices'] for e in by_video] == [[0, 1, 3, 4, 5], [2]]
    assert [e['query_indices'] for e in by_record] == [[0, 1], [2], [3, 4, 5]]
    for episodes in (by_video, by_record):
        assert sorted(i for e in episodes for i in e['query_indices']) == list(range(6))
    assert len({e['id'] for e in by_record}) == 3
    assert len({e['video_key'] for e in by_record}) == 2
    with pytest.raises(ValueError, match='No turns were truncated'):
        group_episodes(rows, [100]*6, [20]*6, 150, 'video')
    assert [e['tokens'] for e in group_episodes(rows, [100]*6, [20]*6, 192, 'video')] == [180, 100]


@pytest.mark.parametrize('corruption', ['missing_turn', 'reorder', 'different_video', 'different_duration', 'duplicate_id'])
def test_reject_corrupt_record_memberships(corruption):
    rows = grouping_rows()
    if corruption == 'missing_turn': rows[1]['turn_index'] = 2
    if corruption == 'reorder': rows[:2] = rows[1::-1]
    if corruption == 'different_video': rows[1]['video_key'] = 'other'
    if corruption == 'different_duration': rows[1]['duration'] = 11
    if corruption == 'duplicate_id': rows[1]['id'] = rows[0]['id']
    with pytest.raises(ValueError): group_memberships(rows, 'record')


def test_record_mode_requires_provenance():
    rows = grouping_rows()
    del rows[0]['source_id']
    with pytest.raises(ValueError, match='source_id'): group_memberships(rows, 'record')
    with pytest.raises(ValueError): group_memberships(rows, 'unsupported')


@pytest.mark.parametrize('counts,frames', [((4,), 3), ((4, 2, 5), 4), ((None, 4, None), 7)])
def test_native_multiturn_template_and_no_answer_history(processor, counts, frames):
    item = make_episode(processor, counts, frames)
    record = record_for(frames)
    messages = []
    for i, row in enumerate(item['rows']):
        content = [dict(type='video', video='provided_in_memory')] if i == 0 else []
        content.append(dict(type='text', text=suffix(row, timestamp_texts=timestamp_texts(processor, record))))
        messages.append(dict(role='user', content=content))
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    texts, _ = processor.get_text_with_replacements([text], [], [_replacement(processor, record, fast=False)], [])
    native = processor(text=texts, return_tensors='pt')
    for key in ('input_ids', 'attention_mask', 'mm_token_type_ids'):
        torch.testing.assert_close(item['inputs'][key], native[key], atol=0, rtol=0)
    assert int((native.input_ids == processor.video_token_id).sum()) == record['tokens']
    changed = copy.deepcopy(item['rows'])
    for row in changed:
        if row['task'] == 'choice': row['answer_index'] = (row['answer_index']+1) % len(row['options'])
        else: row['spans'] = [[1., 1.5]]
    for key, value in episode_inputs(processor, record, changed).items():
        torch.testing.assert_close(value, item['inputs'][key], atol=0, rtol=0)


def test_one_turn_is_original_qa_batch(processor):
    episode = make_episode(processor, (4,))
    item = dict(row=episode['rows'][0], inputs=episode['inputs'], times=episode['times'],
                labels=episode['labels'][0], audit=episode['audit'])
    a = MultiturnCollator(processor.tokenizer, 4096)([episode])
    b = PackedTimestampOptionCollator(processor.tokenizer, 4096)([item])
    for key in ('input_ids', 'mm_token_type_ids', 'video_grid_thw', 'visual_features',
                'pair_indices', 'query_indices', 'labels', 'score_mask', 'decision_tasks'):
        torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)


def test_ragged_readouts_loss_and_empty_packs(processor):
    items = [make_episode(processor), make_episode(processor, (3, None), 6, 'other')]
    batch = MultiturnCollator(processor.tokenizer, 4096)(items)
    assert batch['sample_count'] == 5 and batch['episode_count'] == 2
    assert batch['turn_episode_indices'].tolist() == [0, 0, 0, 1, 1]
    assert batch['score_mask'].sum(1).tolist() == [4, 2, 5, 3, 3]
    assert batch['decision_tasks'].tolist() == [1, 1, 1, 1, 0]
    logits = torch.randn(5, 5, requires_grad=True)
    out = typed_outputs(logits, batch['score_mask'], batch['decision_tasks'], batch['labels'])
    reference = torch.stack([typed_outputs(logits[i:i+1], batch['score_mask'][i:i+1],
        batch['decision_tasks'][i:i+1], batch['labels'][i:i+1])['loss'] for i in range(5)]).mean()
    torch.testing.assert_close(out['loss'], reference)
    out['loss'].backward()
    assert not logits.grad[~batch['score_mask']].any()
    empty = MultiturnCollator(processor.tokenizer, 4096)([dict(empty_pack=True)])
    assert empty['sample_count'] == 0 and empty['episode_lengths'] == [] and not empty['score_mask'].any()
    zeros = typed_outputs(torch.zeros_like(empty['labels']), empty['score_mask'], empty['decision_tasks'], empty['labels'])
    assert zeros['loss'] == 0 and torch.isfinite(zeros['probabilities']).all()


def test_readouts_reject_option_migration(processor):
    item = make_episode(processor)
    with pytest.raises(ValueError, match='boundaries'):
        episode_readouts(processor.tokenizer, item['inputs']['input_ids'][0], [3, 3, 5])


def test_one_feature_read_for_all_turns(processor):
    sample = make_episode(processor)
    dataset = object.__new__(MultiturnDataset)
    dataset.rows = sample['rows']; dataset.processor = processor
    dataset.episodes = [dict(id='v', query_indices=[0, 1, 2], tokens=sample['inputs']['input_ids'].shape[1])]
    dataset.feature_variant_root = None; dataset.feature_root = '/unused'
    dataset.fps = 2.; dataset.feature_spec = dict(cache_id='test')
    record = record_for()
    with patch('timejev.features.read_features', return_value=(sample['inputs']['visual_features'], record)) as read:
        actual = dataset[0]
    assert read.call_count == 1 and len(actual['labels']) == 3
    assert [label.argmax().item() for label in actual['labels']] == [0, 1, 2]


