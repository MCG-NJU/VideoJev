"""Check public multi-query routes without GPU execution or benchmark downloads."""
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_helpers import record_for
from timejev.decision_inputs import load_processor
from timejev.multiturn_evaluation import predict_video, query_chunks
from timejev.multiturn_inputs import episode_inputs


@pytest.fixture(scope='module')
def processor():
    path = os.environ.get('VIDEOJEV_TEST_MODEL', 'models/Qwen3.5-4B')
    if not Path(path).is_dir():
        pytest.skip('Set VIDEOJEV_TEST_MODEL to a local Qwen3.5-4B tokenizer')
    return load_processor(path)


@pytest.mark.parametrize('task', ['choice', 'grounding'])
def test_complete_query_groups_preserve_tokens_and_features(processor, task):
    record = record_for()
    rows = [dict(id=f'v/{i}', video_key='v', video_path='/example.mp4', duration=2.,
                 task=task, query=f'What happens in event {i}?', spans=[[0., 1.]])
            for i in range(4)]
    if task == 'choice':
        for row in rows:
            row.update(options=['A. First.', 'B. Second.'], answer_index=1)
    features = torch.randn(record['tokens'], 8, dtype=torch.bfloat16)
    native = episode_inputs(processor, record, rows)
    calls = []

    class Model:
        def __call__(self, **batch):
            assert 'labels' not in batch
            assert batch['episode_lengths'] == [native['input_ids'].shape[1]]
            assert batch['turn_episode_indices'].tolist() == [0] * len(rows)
            assert torch.equal(batch['input_ids'][:, :native['input_ids'].shape[1]], native['input_ids'])
            assert torch.equal(batch['visual_features'], features)
            calls.append(1)
            logits = torch.zeros_like(batch['score_mask'], dtype=torch.float32)
            return dict(logits=logits, probabilities=logits.sigmoid())

    result = predict_video(Model(), processor, features, record, rows, 4096, 'cpu')
    assert len(calls) == result['forward_calls'] == 1
    assert [row['id'] for row in result['predictions']] == [row['id'] for row in rows]
    with pytest.raises(ValueError, match='Single intact query'):
        query_chunks(processor, record, rows, 64)
    first_length = episode_inputs(processor, record, rows[:1])['input_ids'].shape[1]
    chunks = query_chunks(processor, record, rows, (first_length + 63) // 64 * 64)
    assert len(chunks) > 1
    assert [i for chunk in chunks for i in range(chunk['start'], chunk['end'])] == list(range(len(rows)))


def test_rgb_multiturn_reuses_already_encoded_features(monkeypatch):
    from timejev.rgb_model import RGBMultiturnTimeJev
    from timejev.multiturn_model import MultiturnTimeJev
    features = torch.ones(3, 8)
    model = RGBMultiturnTimeJev.__new__(RGBMultiturnTimeJev)
    torch.nn.Module.__init__(model)

    def unexpected_encode(*args, **kwargs):
        pytest.fail('A query chunk must reuse the already encoded complete video')

    model.backbone = SimpleNamespace(visual=unexpected_encode)
    monkeypatch.setattr(MultiturnTimeJev, 'forward', lambda self, **batch: batch)
    output = model.forward(episode_lengths=[128], visual_features=features,
                           video_grid_thw=torch.tensor([[1, 2, 6]]), feature_cache_ids=['test'])
    assert output['visual_features'] is features


def test_portable_paths():
    from timejev.config import ROOT, normalize_recipe
    value = normalize_recipe(dict(train='data/joint/train.jsonl'))
    assert value['train'] == str(ROOT/'data/joint/train.jsonl')


def test_qa_context_selection_uses_primary_metric_and_keeps_ties():
    from timejev.qa_eval_settings import select_context_results
    comparison = {str(c): {name: dict(accuracy=.6, m_avg=.5, n=2)
        for name in ('lvbench', 'videomme', 'longvideobench', 'mlvu')}
        for c in (196608, 262144)}
    comparison['262144']['mlvu'].update(accuracy=.4, m_avg=.7)
    comparison['262144']['lvbench']['accuracy'] = .8
    selected = select_context_results(comparison)
    assert selected['mlvu']['context'] == selected['lvbench']['context'] == 262144
    assert selected['videomme']['context'] == selected['longvideobench']['context'] == 196608
    assert selected['mlvu']['metrics'] is comparison['262144']['mlvu']
