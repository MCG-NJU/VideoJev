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
