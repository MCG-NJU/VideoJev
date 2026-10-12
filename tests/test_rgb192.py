import os
import json
from pathlib import Path

import pytest
import torch

from timejev.rgb_data import CAPACITY, MIN_PIXELS, geometry, plan_conversation, text_lengths
from timejev.rgb_media import metadata, timeline


def test_compress_before_split_and_keep_each_question_once():
    record = dict(frames=2048, original_hw=[720, 1280])
    pieces, dropped = plan_conversation(record, [10000]*8, [15000]*8)
    assert len(pieces) > 1 and not dropped
    assert [i for p in pieces for i in p['members']] == list(range(8))
    for piece in pieces:
        assert piece['tokens'] <= CAPACITY
        assert MIN_PIXELS <= piece['geometry']['pixels_per_frame'] <= 640*640
    whole, dropped = plan_conversation(record, [10000]*3, [15000]*3)
    assert len(whole) == 1 and whole[0]['members'] == [0, 1, 2] and not dropped


def test_drop_only_impossible_single_question_and_continue():
    record = dict(frames=2048, original_hw=[720, 1280])
    pieces, dropped = plan_conversation(record, [500, CAPACITY+1, 700], [500, CAPACITY+1, 700])
    assert dropped == [1]
    assert [i for p in pieces for i in p['members']] == [0, 2]


def test_eval_may_go_below_train_pixel_floor():
    record = dict(frames=2048, original_hw=[720, 1280])
    assert geometry(record, 120000) is None
    shape = geometry(record, 120000, minimum=1024)
    assert shape and shape['pixels_per_frame'] < MIN_PIXELS
    assert shape['tokens']+120000 <= CAPACITY


@pytest.fixture(scope='module')
def processor():
    from timejev.decision_inputs import load_processor
    model = os.environ.get('VIDEOJEV_TEST_MODEL', 'models/Qwen3.5-4B')
    if not Path(model).is_dir():
        pytest.skip('Set VIDEOJEV_TEST_MODEL to a local Qwen3.5-4B model')
    return load_processor(model)


@pytest.mark.parametrize('name', ['vtg0', 'vtg1', 'qa'])
def test_exact_budget_matches_real_native_mixed_turn_input(processor, name):
    if not (Path(__file__).resolve().parents[1]/'data/head_only/train.jsonl').exists():
        pytest.skip('Required prepared data is not bundled with the source release')
    from timejev.multiturn_inputs import episode_inputs
    root = Path(__file__).resolve().parents[1]
    with (root/f'data/head_only/{name}.jsonl').open() as f:
        row = json.loads(next(f))
    # Native first/continuation prompts of BOTH tasks in one video episode.
    choice = dict(row, id='choice', task='choice', query='Which action happens next?',
                  options=['A. First action', 'B. Second action'], answer_index=1)
    grounding = dict(row, id='grounding', task='grounding', query='A person enters.', spans=[[0., 2.]])
    rows = [choice, grounding, dict(choice, id='choice2')]
    record = dict(video_key=row['video_key'], frames=19, fps=2., effective_fps=2.,
                  sampled_duration=9.4, original_hw=[360, 640])
    first, cont = text_lengths(processor, record, rows)
    pieces, dropped = plan_conversation(record, first, cont)
    assert not dropped and len(pieces) == 1
    spec = pieces[0]
    actual = episode_inputs(processor, timeline(dict(record, **spec['geometry'])), rows)
    assert actual['input_ids'].shape[1] == spec['tokens']


def test_rgb_collator_handles_ragged_choice_and_grounding(processor):
    from timejev.rgb_data import text_item
    from timejev.multiturn_data import MultiturnCollator
    row = dict(id='v/0', video_key='v', video_path='/unused.mp4', duration=4.,
               query='Which event?', spans=[[0., 2.]])
    choice = dict(row, id='v/1', task='choice', options=['A. a', 'B. b', 'C. c'], answer_index=2)
    record = dict(video_key='v', frames=8, fps=2., effective_fps=2., sampled_duration=4.,
                  original_hw=[360, 640], source=dict(kind='frames_4fps', path='/frames'))
    first, cont = text_lengths(processor, record, [row, choice])
    pieces, _ = plan_conversation(record, first, cont)
    item = text_item(processor, [row, choice], record, dict(id='v/part0', **pieces[0]))
    batch = MultiturnCollator(processor.tokenizer, CAPACITY, include_score_durations=True)([item])
    assert batch['sample_count'] == 2 and batch['episode_count'] == 1
    assert batch['score_mask'].sum(1).tolist() == [4, 3]
    assert batch['decision_tasks'].tolist() == [0, 1]
    assert batch['score_durations'][0].sum() == 4
    assert 'visual_features' not in batch and 'video_pixel_chunks' in batch


def test_native_chunked_preprocessing_matches_whole(processor):
    if not (Path(__file__).resolve().parents[1]/'data/frames/lvbench/00000000/.complete.json').exists():
        pytest.skip('Required prepared data is not bundled with the source release')
    from timejev.rgb_media import preprocess, sampled_chunks
    from transformers.image_utils import SizeDict
    row = dict(video_key='lvbench/Cm73ma6Ibcs', duration=3665.5,
               video_path='data/videos/lvbench/00000000.mp4')
    record = metadata(row, 'lvbench')
    record['frames'] = 5
    record.update(geometry(record, 100))
    chunks = preprocess(processor, record, chunk_frames=2)
    frames = next(sampled_chunks(record, chunk_frames=6))
    vp = processor.video_processor
    frames = vp.resize(frames, size=SizeDict(height=record['processed_hw'][0],
                                           width=record['processed_hw'][1]), resample=vp.resample)
    whole = vp(videos=[frames], do_resize=False, do_sample_frames=False, return_tensors='pt')
    torch.testing.assert_close(torch.cat(chunks), whole['pixel_values_videos'].to(torch.bfloat16), rtol=0, atol=0)
