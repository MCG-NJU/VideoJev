"""Production and mixed-task prediction use bounded caches only when supported."""
from contextlib import contextmanager
import numpy as np
import pytest
import torch

from timejev.engine import predict
from timejev.decision_benchmark import predict_decisions


class Predictor:
    def __init__(self, fail=False):
        self.training = True
        self.fail = fail

    def eval(self):
        self.training = False
        return self

    def __call__(self, **inputs):
        assert not self.training and not torch.is_grad_enabled()
        assert 'labels' not in inputs
        if self.fail:
            raise RuntimeError('test prediction failure')
        logits = torch.tensor([[1., -1.]])
        return dict(logits=logits, probabilities=logits.sigmoid())


class CachedPredictor(Predictor):
    active = False
    entered = 0

    @contextmanager
    def video_cache(self, max_tokens):
        assert max_tokens == 147456
        assert not self.training and not torch.is_grad_enabled()
        self.active = True
        self.entered += 1
        try:
            yield {}
        finally:
            self.active = False

    def __call__(self, **inputs):
        assert self.active
        return super().__call__(**inputs)


def batch():
    return dict(rows=[dict(id='one', sampled_duration=2.)],
        times=[np.array([.5, 1.5])], audits=[dict(score_bounds=[[0., 1.], [1., 2.]])],
        score_mask=torch.ones((1, 2), dtype=torch.bool), labels=torch.tensor([[1., 0.]]))


@pytest.mark.parametrize('predictor', [predict, predict_decisions])
def test_cached_and_default_predict_are_equal(predictor):
    plain, cached = Predictor(), CachedPredictor()
    expected = predictor(plain, [batch(), batch()], 'cpu')
    actual = predictor(cached, [batch(), batch()], 'cpu')
    assert actual == expected
    assert cached.entered == 1 and not cached.active


@pytest.mark.parametrize('predictor', [predict, predict_decisions])
def test_prediction_failure_releases_cache(predictor):
    model = CachedPredictor(fail=True)
    with pytest.raises(RuntimeError, match='test prediction failure'):
        predictor(model, [batch()], 'cpu')
    assert not model.active
