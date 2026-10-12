import os
import json
from pathlib import Path

import pytest
import torch

from timejev.context_eval import (CONTEXTS, COUNTS, assignments, choose_geometry,
    question_item, validate_spec, visual_budget)
from timejev.data import read_jsonl

ROOT = Path(__file__).resolve().parents[1]


def test_visual_budget_and_full_question_are_independent_constraints():
    record = dict(frames=2048, original_hw=[1080, 1920])
    shapes = [choose_geometry(record, 12000, context) for context in CONTEXTS]
    assert [visual_budget(c) for c in CONTEXTS] == [157286, 209715]
    assert shapes[0]['tokens'] < shapes[1]['tokens']
    for context, shape in zip(CONTEXTS, shapes):
        assert shape['grid_thw'][0] == 1024
        assert shape['tokens'] <= visual_budget(context)
        assert shape['pixels_per_frame'] <= 640*640
        tight = choose_geometry(record, context-30000, context)
        assert tight['tokens'] <= 30000 and tight['tokens'] < shape['tokens']
    with pytest.raises(ValueError, match='cannot fit'):
        choose_geometry(record, CONTEXTS[0], CONTEXTS[0])


@pytest.fixture(scope='module')
def plans():
    if not (ROOT/'data/context_eval_v1/ready.json').exists():
        pytest.skip('Prepare both context plans to run this integration test')
    return {c: json.loads((ROOT/f'data/context_eval_v1/context_{c}.json').read_text()) for c in CONTEXTS}


def test_complete_both_plans_and_56_rank_assignment(plans):
    for context, plan in plans.items():
        total = 0
        for name in COUNTS:
            spec = plan['benchmarks'][name]
            rows = read_jsonl(spec['manifest'])
            validate_spec(spec, rows, context)
            assert len(rows) == COUNTS[name]
            shards = assignments(spec, 56)
            assert sorted(i for shard in shards for i in shard) == list(range(len(spec['videos'])))
            total += len(rows)
        assert total == 7760
    for name in COUNTS:
        left, right = (plans[c]['benchmarks'][name]['videos'] for c in CONTEXTS)
        assert [v['record'] for v in left] == [v['record'] for v in right]


def test_native_token_lengths_and_audits_for_both_contexts(plans):
    from timejev.decision_inputs import load_processor
    processor = load_processor(os.environ.get('VIDEOJEV_TEST_MODEL', 'models/Qwen3.5-4B'))
    for context, plan in plans.items():
        for name in COUNTS:
            spec = plan['benchmarks'][name]
            rows = read_jsonl(spec['manifest'])
            video = max(spec['videos'], key=lambda v: max(q['tokens'] for q in v['questions']))
            q = max(video['questions'], key=lambda q: q['tokens'])
            item = question_item(processor, rows[q['query_index']], video, q, context)
            assert item['inputs']['input_ids'].shape[1] == q['tokens']
            assert item['audit']['max_sequence_length'] == context
            assert item['audit']['visual_token_budget'] == visual_budget(context)


def test_four_benchmarks_two_contexts_export_and_resume(tmp_path, monkeypatch, plans):
    from scripts import evaluate_contexts as evaluate, pipeline
    config = json.loads((ROOT/'configs/joint.json').read_text())
    checkpoint = tmp_path/'model.pt'
    torch.save(dict(config=config, checkpoint_id='mock-contexts-only', step=1, model={}), checkpoint)
    calls = []
    monkeypatch.setattr(evaluate, 'setup', lambda seed: (0, 1, 'cpu'))
    monkeypatch.setattr(evaluate, 'load_rgb_model', lambda *a: object())
    monkeypatch.setattr(evaluate, 'load_processor', lambda *a: None)
    monkeypatch.setattr(evaluate, 'make_loader', lambda dataset, *a, **kw: dataset)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    def predict(model, dataset, spec, rows, config, device, processor, *, on_video):
        for index in dataset.indices:
            video = spec['videos'][index]
            calls.append(video['video_key'])
            predictions = []
            for q in video['questions']:
                row = rows[q['query_index']]
                values = [float(i == row['answer_index']) for i in range(len(row['options']))]
                predictions.append(dict(id=row['id'], task='choice', answer_index=row['answer_index'],
                    options=row['options'], probabilities=values, logits=values))
            on_video(video, dict(predictions=predictions, cache=dict(seconds=1)))
        return [], []
    monkeypatch.setattr(evaluate, 'predict_videos', predict)
    output = tmp_path/'evaluation'
    monkeypatch.setattr('sys.argv', ['evaluate', '--checkpoint', str(checkpoint),
        '--plans', str(ROOT/'data/context_eval_v1'), '--out', str(output)])
    evaluate.main()
    assert len(calls) == 2*2878
    assert pipeline.verify_completion(output)['questions_per_context'] == 7760
    summary = json.loads((output/'summary.json').read_text())
    for context in CONTEXTS:
        for name in COUNTS:
            assert summary[str(context)][name]['m_avg' if name == 'mlvu' else 'accuracy'] == 1.
    calls.clear()
    evaluate.main()
    assert calls == []
    # An interrupted benchmark resumes only its missing video, even if sharding changes.
    target = output/'context_262144/mlvu'
    (target/'completion.json').unlink()
    video = plans[262144]['benchmarks']['mlvu']['videos'][0]
    evaluate.video_path(target, video).unlink()
    evaluate.main()
    assert calls == [video['video_key']]
    # Corruption is rejected, never silently treated as an acceptable result.
    (target/'metrics.json').write_text('{}')
    with pytest.raises(ValueError, match='artifact changed'):
        evaluate.main()


def test_video_result_integrity(tmp_path):
    from scripts.evaluate_contexts import read_video
    from timejev.context_eval import digest_json
    from timejev.engine import save_json
    video = dict(video_key='v', questions=[dict(id='q')])
    identity = dict(checkpoint='one', context=262144)
    payload = dict(predictions=[dict(id='q', logits=[.2, .3])])
    path = tmp_path/'video.json'
    save_json(dict(identity=identity, video_key='v', payload=payload,
                   payload_sha256=digest_json(payload)), path)
    assert read_video(path, identity, video) == payload
    with pytest.raises(ValueError, match='changed'):
        read_video(path, dict(identity, checkpoint='two'), video)
    bad = json.loads(path.read_text())
    bad['payload']['predictions'][0]['logits'][0] = 100
    save_json(bad, path)
    with pytest.raises(ValueError, match='changed'):
        read_video(path, identity, video)


def test_source_snapshot_reuse_and_mutation_guard(tmp_path):
    from scripts.context_eval_pipeline import snapshot
    from scripts.pipeline import digest, save
    project, plans, out = (tmp_path/name for name in ('project', 'plans', 'run'))
    for name in ('timejev', 'scripts', 'configs', '.deps'):
        (project/name).mkdir(parents=True)
    for name in ('timejev/input.py', 'scripts/eval.py', 'configs/eval.json',
                 'sitecustomize.py', 'requirements-overlay.txt'):
        (project/name).write_text('original')
    plans.mkdir()
    save(dict(code={'timejev/input.py': digest(project/'timejev/input.py')}), plans/'identity.json')
    save(dict(complete=True, files={'identity.json': digest(plans/'identity.json')}), plans/'ready.json')
    checkpoint = tmp_path/'model.pt'
    checkpoint.write_text('checkpoint')
    source = snapshot(project, out, plans, checkpoint)
    # Live repo edits cannot affect a running/restarted evaluation snapshot.
    (project/'scripts/eval.py').write_text('later edit')
    assert snapshot(project, out, plans, checkpoint) == source
    assert (source/'scripts/eval.py').read_text() == 'original'
    (source/'scripts/eval.py').write_text('changed snapshot')
    with pytest.raises(ValueError, match='snapshot changed'):
        snapshot(project, out, plans, checkpoint)
