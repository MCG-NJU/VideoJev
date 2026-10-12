"""Public prediction results expose probabilities and preserve task decisions."""
from types import SimpleNamespace

import pytest
import torch

from videojev_hf.processing_videojev import format_predictions


def test_prediction_results_omit_logits_and_padding():
    rows = [dict(id='span', task='grounding'),
            dict(id='qa', task='choice', options=['A. Pasta.', 'B. Rice.'])]
    logits = torch.tensor([[2., 2., -2.], [-2., 2., 100.]])
    output = SimpleNamespace(
        logits=logits,
        probabilities=torch.tensor([[.9, .8, .1], [.1, .9, 0.]]),
        score_mask=torch.tensor([[True, True, True], [True, True, False]]))
    record = dict(score_times=[.5, 1.5, 2.5],
                  score_bounds=[[0., 1.], [1., 2.], [2., 3.]], sampled_duration=3.)
    decoder = dict(threshold=.5, smooth=1, postprocess='round_repair_clamp_v1')
    results = format_predictions(output, rows, record, decoder)
    assert set(results[0]) == {'id', 'task', 'probabilities', 'interval'}
    assert set(results[1]) == {'id', 'task', 'probabilities', 'answer_index', 'answer'}
    assert [r['id'] for r in results] == ['span', 'qa']
    assert results[0]['interval'] == [0., 2.]
    assert results[0]['probabilities'] == pytest.approx([.9, .8, .1])
    assert results[1]['probabilities'] == pytest.approx([.1, .9])
    assert results[1]['answer_index'] == 1
    assert results[1]['answer'] == 'B. Rice.'


def test_decoder_config_defaults_and_partial_override_roundtrip(tmp_path):
    from videojev_hf.configuration_videojev import VideoJevConfig
    config = VideoJevConfig()
    assert config.decoder == dict(threshold=.25, smooth=1, postprocess='round_repair_clamp_v1')
    custom = VideoJevConfig(decoder={'threshold': .4})
    assert custom.decoder == dict(threshold=.4, smooth=1, postprocess='round_repair_clamp_v1')
    custom.save_pretrained(tmp_path)
    assert VideoJevConfig.from_pretrained(tmp_path).decoder == custom.decoder
    assert config.decoder['threshold'] == .25


@pytest.mark.parametrize('threshold', [0., 1., -.1, float('nan')])
def test_invalid_decoder_config_rejected(threshold):
    from videojev_hf.configuration_videojev import VideoJevConfig
    with pytest.raises(ValueError, match='threshold'):
        VideoJevConfig(decoder={'threshold': threshold})


@pytest.mark.parametrize('api', ['predict_features', 'predict_video'])
def test_call_threshold_changes_only_grounding_without_mutating_default(monkeypatch, api):
    from videojev_hf import modeling_videojev as module
    logits = torch.tensor([[-2., -.8, .8, -.8, -2.], [-1., 1., 0., 0., 0.]])
    output = SimpleNamespace(logits=logits, probabilities=torch.stack([
        logits[0].sigmoid(), torch.tensor([.1, .9, 0., 0., 0.])]),
        score_mask=torch.tensor([[True]*5, [True, True, False, False, False]]))
    record = dict(score_times=[.5, 1.5, 2.5, 3.5, 4.5], sampled_duration=5.,
                  score_bounds=[[float(i), float(i+1)] for i in range(5)], grid_thw=[5, 2, 2])
    rows = [dict(id='span', task='grounding'),
            dict(id='qa', task='choice', options=['A. Pasta.', 'B. Rice.'])]
    monkeypatch.setattr(module, 'prepare_query_groups', lambda *args: iter([({}, rows)]))
    monkeypatch.setattr(module, 'prepare_video', lambda *args, **kwargs: (None, record))

    class Predictor:
        predict_video = module.VideoJevForVideoUnderstanding.predict_video
        predict_features = module.VideoJevForVideoUnderstanding.predict_features
        config = SimpleNamespace(decoder=dict(threshold=.25, smooth=1, postprocess='round_repair_clamp_v1'))

        def encode_video(self, *args):
            return None

        def __call__(self, **kwargs):
            return output

    model = Predictor()
    method = getattr(model, api)
    args = (None, None, record, rows) if api == 'predict_features' else (None, 'demo.mp4', rows)
    default = method(*args)
    overridden = method(*args, threshold=.5)
    assert default[0]['interval'] == [1., 4.]
    assert overridden[0]['interval'] == [2., 3.]
    assert default[0]['probabilities'] == overridden[0]['probabilities']
    assert default[1] == overridden[1]
    assert model.config.decoder['threshold'] == .25
    assert method(*args) == default
    model.config.decoder['threshold'] = .5
    assert method(*args) == overridden
    for invalid in [0., 1., -.1, float('nan')]:
        with pytest.raises(ValueError, match='threshold'):
            method(*args, threshold=invalid)
