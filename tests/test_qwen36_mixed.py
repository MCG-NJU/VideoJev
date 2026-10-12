import json
from pathlib import Path

import pytest
import torch

from timejev.qwen36_data import evaluation_schedule
from timejev.qwen36_recipe import parameter_groups, validate_recipe

ROOT = Path(__file__).resolve().parents[1]


def test_collective_schedule_covers_questions_and_keeps_videos_on_one_rank():
    spec = dict(videos=[dict(questions=[dict(tokens=n) for n in lengths])
                       for lengths in ([600, 600, 500], [200]*7, [700, 300], [100]*10)])
    seen, owners = [], {}
    for rank in range(5):
        videos, packs = evaluation_schedule(spec, 1024, rank, 5)
        for pack in packs:
            assert sum(spec['videos'][i]['questions'][j]['tokens'] for i, j in pack) <= 1024
            seen.extend(pack)
            for i, _ in pack:
                assert owners.setdefault(i, rank) == rank
        assert set(videos) == {i for pack in packs for i, _ in pack}
    assert sorted(seen) == [(i, j) for i, video in enumerate(spec['videos']) for j in range(len(video['questions']))]


def test_overlength_question_is_never_truncated():
    with pytest.raises(ValueError, match='exceeds'):
        evaluation_schedule(dict(videos=[dict(questions=[dict(tokens=1025)])]), 1024, 0, 1)


def test_router_uses_llm_lr_and_zero_placeholder_uses_original_shape_for_decay():
    model = torch.nn.Module()
    model.router = torch.nn.Linear(4, 4)
    model.head = torch.nn.Linear(4, 2)
    model.router.weight.ds_shape = (4, 4)
    model.router.weight.data = torch.empty(0)
    config = dict(lr=2e-6, head_lr=1e-4, weight_decay=.01)
    groups = parameter_groups(model, config)
    by_parameter = {id(p): g for g in groups for p in g['params']}
    assert by_parameter[id(model.router.weight)]['lr'] == 2e-6
    assert by_parameter[id(model.router.weight)]['weight_decay'] == .01
    assert by_parameter[id(model.router.bias)]['weight_decay'] == 0
    assert by_parameter[id(model.head.weight)]['lr'] == 1e-4


@pytest.mark.parametrize('stage', ['head_only', 'joint'])
def test_fixed_recipe_rejects_accidental_lr_or_loss_changes(stage):
    config = json.loads((ROOT/f'configs/qwen36_{stage}.json').read_text())
    validate_recipe(config)
    config['soft_iou_weight'] = 2. if stage == 'head_only' else 0.
    with pytest.raises(ValueError, match='soft_iou_weight'):
        validate_recipe(config)


def test_fingerprint_detects_same_size_export_corruption(tmp_path):
    from scripts.evaluate_qwen36 import checkpoint_fingerprint
    (tmp_path/'weight.bin').write_bytes(b'abc')
    (tmp_path/'complete_manifest.json').write_text(json.dumps(dict(complete=True,
        files={'weight.bin': 3}, head_only=False)))
    before = checkpoint_fingerprint(tmp_path)
    (tmp_path/'weight.bin').write_bytes(b'abd')
    assert checkpoint_fingerprint(tmp_path) != before


def test_all_seven_exports_resume_and_corruption_guard(tmp_path, monkeypatch):
    if not (Path(__file__).resolve().parents[1]/'data/qwen36_mixed_v1/ready.json').exists():
        pytest.skip('Required prepared data is not bundled with the source release')
    import deepspeed
    from scripts import evaluate_qwen36 as evaluate
    from scripts.pipeline import verify_completion
    from timejev.engine import atomic_save, save_json
    config = json.loads((ROOT/'configs/qwen36_joint.json').read_text())
    checkpoint = tmp_path/'checkpoint'
    checkpoint.mkdir()
    save_json(dict(recipe=config, checkpoint_id='mock-mixed35b', step=375), checkpoint/'task_model_config.json')
    save_json(dict(complete=True, files={'task_model_config.json': (checkpoint/'task_model_config.json').stat().st_size}),
              checkpoint/'complete_manifest.json')
    monkeypatch.setattr(evaluate, 'setup', lambda seed: (0, 64, 'cpu'))
    for name in ('broadcast_object_list', 'all_reduce', 'barrier', 'destroy_process_group'):
        monkeypatch.setattr(evaluate.dist, name, lambda *a, **k: None)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(evaluate, 'build_grounder', lambda *a, **k: object())
    monkeypatch.setattr(evaluate, 'build_frozen_vision', lambda *a, **k: object())
    monkeypatch.setattr(evaluate, 'load_exported_task_checkpoint', lambda *a, **k: None)
    class Engine:
        def eval(self):
            return self
    monkeypatch.setattr(deepspeed, 'initialize', lambda *a, **k: (Engine(), None, None, None))
    calls = []
    def predict(engine, visual, spec, rows, config, capacity, rank, world, device, processor, target):
        calls.append((target.name, capacity))
        for worker in range(1, world):
            atomic_save([], target/f'rank{worker}.pt')
            save_json({}, target/f'cache_rank{worker}.json')
        predictions = []
        for row in rows:
            if row.get('task') == 'choice':
                values = [float(i == row['answer_index']) for i in range(len(row['options']))]
                predictions.append(dict(id=row['id'], task='choice', answer_index=row['answer_index'],
                    options=row['options'], probabilities=values, logits=values))
            else:
                predictions.append(dict(id=row['id'], times=[row['duration']/2], logits=[1.],
                    bounds=[[0., row['duration']]], sampled_duration=row['duration'], audit={}))
        return predictions, {}
    monkeypatch.setattr(evaluate, 'predict_packs', predict)
    monkeypatch.setattr(evaluate, 'select_decoder', lambda *a, **k: dict(threshold=.5, smooth=1,
        postprocess='round_repair_clamp_v1'))
    output = tmp_path/'evaluation'
    monkeypatch.setattr('sys.argv', ['evaluate', '--checkpoint', str(checkpoint), '--plans',
        str(ROOT/'data/qwen36_mixed_v1'), '--out', str(output)])
    evaluate.main()
    assert len(calls) == 7
    assert len(verify_completion(output)['benchmarks']) == 7
    summary = json.loads((output/'summary.json').read_text())
    for context in (262144,):
        for name in evaluate.QA_NAMES:
            assert summary[f'{name}_{context}']['m_avg' if name == 'mlvu' else 'accuracy'] == 1.
    calls.clear()
    evaluate.main()
    assert calls == []
    (output/'qa_262144/mlvu/metrics.json').write_text('{}')
    with pytest.raises(ValueError, match='artifact changed'):
        evaluate.main()
