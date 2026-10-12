"""Run the RGB seven-benchmark export/reuse contract with mocked GPU predictions."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import evaluate_rgb192 as evaluate, pipeline
from timejev.engine import sha256

ROOT = Path(__file__).resolve().parents[1]


def test_rgb_seven_benchmarks_and_verified_reuse(tmp_path, monkeypatch):
    if not (Path(__file__).resolve().parents[1]/'data/benchmarks/charades/questions.jsonl').exists():
        pytest.skip('Required prepared data is not bundled with the source release')
    c = json.loads((ROOT/'configs/rgb192_joint.json').read_text())
    plan = dict(benchmarks={name: dict(manifest=str(ROOT/f'data/benchmarks/{name}/questions.jsonl'),
        manifest_sha256=sha256(ROOT/f'data/benchmarks/{name}/questions.jsonl'), videos=[])
        for name in evaluate.COUNTS})
    c['rgb_evaluation'] = str(tmp_path/'plan.json')
    (tmp_path/'plan.json').write_text(json.dumps(plan))
    checkpoint = tmp_path/'model.pt'
    torch.save(dict(config=c, checkpoint_id='rgb-mock-only', step=1, model={}), checkpoint)
    calls = []
    class Model:
        def load_checkpoint(self, value):
            calls.append('load')
        def to(self, device):
            return self
        def eval(self):
            return self
    monkeypatch.setattr(evaluate, 'setup', lambda seed: (0, 1, 'cpu'))
    monkeypatch.setattr(evaluate, 'from_config', lambda *a, **kw: Model())
    monkeypatch.setattr(evaluate, 'load_processor', lambda *a: SimpleNamespace(tokenizer=None))
    monkeypatch.setattr(evaluate, 'make_loader', lambda dataset, *a, **kw: dataset)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    def predict(model, loader, spec, rows, config, device, processor):
        calls.append('benchmark')
        result = []
        for row in rows:
            if row.get('task') == 'choice':
                values = [float(i == row['answer_index']) for i in range(len(row['options']))]
                result.append(dict(id=row['id'], task='choice', answer_index=row['answer_index'],
                    options=row['options'], probabilities=values, logits=values))
            else:
                result.append(dict(id=row['id'], times=[row['duration']/2], logits=[1.],
                    bounds=[[0., row['duration']]], sampled_duration=row['duration'], audit={}))
        return result, {}
    monkeypatch.setattr(evaluate, 'predict_videos', predict)
    monkeypatch.setattr(evaluate, 'select_decoder', lambda *a: dict(threshold=.5, smooth=1,
                                                                  postprocess='round_repair_clamp_v1'))
    target = tmp_path/'evaluation'
    monkeypatch.setattr('sys.argv', ['evaluate', '--checkpoint', str(checkpoint), '--out', str(target)])
    evaluate.main()
    assert calls == ['load']+['benchmark']*7
    assert pipeline.verify_completion(target)['questions'] == 17164
    summary = json.loads((target/'summary.json').read_text())
    assert set(summary) == pipeline.BENCHMARKS
    assert summary['mlvu']['m_avg'] == 1.
    calls.clear()
    evaluate.main()
    assert not calls
    (target/'lvbench/metrics.json').write_text('{}')
    with pytest.raises(ValueError, match='artifact changed'):
        evaluate.main()
