import json
from pathlib import Path

import pytest

from timejev.hf_evaluation import released_checkpoint_cli, score_predictions
from timejev.user_data import load_manifest, split_stages, write_jsonl


def manifest_rows(tmp_path):
    rows = []
    for video in range(4):
        (tmp_path/f'{video}.mp4').touch()
        for question in range(2):
            rows.append(dict(id=f'{video}/{question}', video_key=str(video),
                video_path=f'{video}.mp4', duration=10., task='choice', query='Which object?',
                options=['A. Cup.', 'B. Book.'], answer_index=question))
    return rows


def test_custom_jsonl_resolves_paths_and_splits_complete_videos(tmp_path):
    original = manifest_rows(tmp_path)
    path = tmp_path/'train.jsonl'
    write_jsonl(original, path)
    rows = load_manifest(path)
    assert all(Path(r['video_path']).is_absolute() for r in rows)
    stages = split_stages(rows)
    assert stages == split_stages(rows)
    assert {r['video_key'] for r in stages['head_only']}.isdisjoint(r['video_key'] for r in stages['joint'])
    assert sorted(r['id'] for v in stages.values() for r in v) == sorted(r['id'] for r in rows)
    for members in stages.values():
        assert members == [r for r in rows if r['id'] in {m['id'] for m in members}]


@pytest.mark.parametrize('field,value', [('answer_index', 2), ('duration', 0), ('query', '<|decide|>')])
def test_custom_jsonl_rejects_invalid_labels_and_inputs(tmp_path, field, value):
    rows = manifest_rows(tmp_path)
    rows[0][field] = value
    path = tmp_path/'train.jsonl'
    write_jsonl(rows, path)
    with pytest.raises(ValueError):
        load_manifest(path)


def test_custom_jsonl_rejects_aliases_crossing_video_split(tmp_path):
    rows = manifest_rows(tmp_path)
    rows[1]['video_key'] = 'alias'
    path = tmp_path/'train.jsonl'
    write_jsonl(rows, path)
    with pytest.raises(ValueError, match='same video_key'):
        load_manifest(path)


def test_rgb_source_inventory_can_be_custom_size(tmp_path):
    from scripts.prepare_rgb192 import source_rows
    stages = split_stages(manifest_rows(tmp_path))
    for stage, rows in stages.items():
        write_jsonl(rows, tmp_path/f'data/{stage}/train.jsonl')
    loaded, bindings = source_rows(tmp_path, 20260924)
    assert loaded == stages
    assert len(bindings) == 2


@pytest.mark.parametrize('script', ['evaluate', 'evaluate_qwen36'])
def test_existing_evaluators_accept_hf_without_training_metadata(tmp_path, monkeypatch, capsys, script):
    import importlib
    checkpoint = tmp_path/'hf'
    checkpoint.mkdir()
    (checkpoint/'config.json').write_text(json.dumps(dict(model_type='videojev')))
    manifest = tmp_path/'eval.jsonl'
    write_jsonl(manifest_rows(tmp_path), manifest)
    argv = ['--checkpoint', str(checkpoint), '--manifest', str(manifest), '--out', str(tmp_path/'out'), '--dry-run']
    assert released_checkpoint_cli(argv)
    monkeypatch.setattr('sys.argv', [script, *argv])
    importlib.import_module('scripts.'+script).main()
    assert json.loads(capsys.readouterr().out)['custom']['qa_context'] == 262144
    (checkpoint/'training_state.json').write_text('{}')
    assert not released_checkpoint_cli(argv)


def test_probability_only_hf_outputs_score_in_manifest_order(tmp_path):
    rows = [dict(id='g', task='grounding', spans=[[2., 4.]]),
            dict(id='q', task='choice', answer_index=1)]
    predictions = [dict(id='q', task='choice', probabilities=[.1, .9], answer_index=1),
                   dict(id='g', task='grounding', probabilities=[.2, .8], interval=[2., 4.])]
    report = score_predictions('custom', predictions, rows, None, tmp_path)
    assert report['grounding']['mIoU'] == 100.
    assert report['choice']['accuracy'] == 1.
    saved = [json.loads(s) for s in (tmp_path/'predictions.jsonl').read_text().splitlines()]
    assert [s['id'] for s in saved] == ['g', 'q']
    assert all('logits' not in s for s in saved)


def test_qa_config_always_uses_fixed_context():
    from timejev.qa_eval_settings import apply_evaluation
    config = dict(training_tasks=['choice'], max_sequence_length=8192)
    evaluation = dict(benchmark='videomme', eval_fps=2., max_frames=2048,
                      feature_root='features', manifest='rows.jsonl', lengths='lengths.json')
    assert apply_evaluation(config, evaluation)['max_sequence_length'] == 262144
    assert config['max_sequence_length'] == 8192


def test_official_qa_scoring_accepts_hf_probability_only_results(tmp_path):
    rows, predictions, questions = [], [], []
    for index in range(1549):
        uid = str(index)
        questions.append(dict(uid=uid, question='Which object?\nA. Cup.\nB. Book.',
                              answer='A', question_type=['object']))
        rows.append(dict(id='lvbench/'+uid, uid=uid, video_key='lvbench/video',
                         query='Which object?', options=['A. Cup.', 'B. Book.'], answer_index=0))
        predictions.append(dict(id='lvbench/'+uid, task='choice', answer_index=0,
                                answer='A. Cup.', probabilities=[.8, .2]))
    reference = tmp_path/'official.jsonl'
    write_jsonl([dict(key='video', qa=questions)], reference)
    report = score_predictions('lvbench', predictions, rows, reference, tmp_path)
    assert report['n'] == 1549 and report['accuracy'] == 1.
    assert 'logits' not in (tmp_path/'predictions.jsonl').read_text()
