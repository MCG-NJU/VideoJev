"""Prepared evaluation plans use the release threshold unless explicitly overridden."""
import hashlib
import json
from pathlib import Path

import pytest

from scripts import prepare_multiturn_eval as prepare


@pytest.mark.parametrize('override, expected', [(None, .25), (.6, .6)])
def test_plan_decoder_default_and_override_preserve_reference(tmp_path, monkeypatch, override, expected):
    monkeypatch.setattr(prepare, 'COUNTS', {'charades': 1})
    reference = tmp_path/'reference'
    (reference/'inputs').mkdir(parents=True)
    cache = tmp_path/'cache'; cache.mkdir()
    (cache/'model.json').write_text(json.dumps({'base': 'example/base'}))
    (cache/'completion.json').write_text('{}')
    manifest = reference/'inputs/questions.jsonl'
    manifest.write_text(json.dumps({'id': 'q', 'video_key': 'v'})+'\n')
    old_decoder = dict(threshold=.15, smooth=1, postprocess='round_repair_clamp_v1')
    (reference/'decoder.json').write_text(json.dumps(old_decoder))
    scope = '147456/charades'
    plan = dict(cache=str(cache), scopes={scope: dict(name='charades', manifest=str(manifest),
                context=147456, task='grounding')},
                jobs=[dict(scope=scope, query_indices=[0], video_key='v', context=147456, task='grounding')])
    (reference/'plan.json').write_text(json.dumps(plan))
    files = {str(p.relative_to(reference)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in reference.rglob('*') if p.is_file()}
    (reference/'identity.json').write_text(json.dumps({'files': files}))
    destination = tmp_path/'prepared/plan.json'
    kwargs = {} if override is None else {'decoder_threshold': override}
    prepare.freeze_plan(reference, destination, **kwargs)
    result = json.loads(destination.read_text())
    assert result['decoder'] == dict(threshold=expected, smooth=1, postprocess='round_repair_clamp_v1')
    assert result['decoder_source_sha256'] == files['decoder.json']
    assert json.loads((reference/'decoder.json').read_text()) == old_decoder
    assert result['jobs'] == plan['jobs']
    for name, digest in result['files'].items():
        assert hashlib.sha256((destination.parent/name).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize('threshold', [0., 1., float('nan')])
def test_invalid_plan_threshold_fails_before_writing(tmp_path, threshold):
    output = tmp_path/'plan.json'
    with pytest.raises(ValueError, match='decoder_threshold'):
        prepare.freeze_plan(tmp_path/'missing', output, decoder_threshold=threshold)
    assert not output.exists()
