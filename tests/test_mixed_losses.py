import math
import pytest
import torch
from torch.nn import functional as F
from timejev.losses import decision_losses
from timejev.decision_inputs import CHOICE, GROUNDING


@pytest.mark.parametrize('stage', ['head_only', 'joint'])
def test_mixed_loss_values_gradients_durations_padding_and_task_weights(stage):
    z = torch.tensor([[1.,3.,-2.,1e5,-1e5], [.4,-.6,.1,1.,-2.], [8.,9.,10.,2.,3.]], requires_grad=True)
    mask = torch.tensor([[1,1,1,0,0], [1,1,1,1,1], [0,0,0,0,0]], dtype=torch.bool)
    labels = torch.tensor([[1.,0,0,0,0], [1.,0.,1.,0.,0.], [0.,0.,0.,0.,0.]])
    tasks = torch.tensor([CHOICE, GROUNDING, CHOICE])
    durations = torch.tensor([[0.,0.,0.,0.,0.], [.1,2.,1.,.5,.4], [0.,0.,0.,0.,0.]])
    weights = dict(soft_iou_weight=2.) if stage=='joint' else {}
    out = decision_losses(z,mask,tasks,labels,score_durations=durations,**weights)
    qa = F.cross_entropy(z[:1,:3],torch.tensor([0]))
    vtg = F.binary_cross_entropy_with_logits(z[1],labels[1])
    if stage=='joint':
        p, y, d = z[1].sigmoid(), labels[1], durations[1]
        vtg = vtg + 2*(1-(d*p*y).sum()/(d*(y+p*(1-y))).sum())
    expected = (qa+vtg)/2
    torch.testing.assert_close(out['loss'],expected)
    actual = torch.autograd.grad(out['loss'],z,retain_graph=True)[0]
    torch.testing.assert_close(actual,torch.autograd.grad(expected,z)[0])
    assert not actual[~mask].any() and torch.isfinite(actual).all()
    assert out['task_counts'].tolist()==[1,1]
    if stage=='head_only':
        assert out['loss_sums'][2:].count_nonzero()==0


def test_empty_rank_has_zero_loss_and_safe_backward():
    z=torch.randn(1,1,requires_grad=True)
    out=decision_losses(z,torch.zeros_like(z,dtype=torch.bool),torch.tensor([GROUNDING]),
        torch.zeros_like(z),soft_iou_weight=2.)
    assert out['loss']==0 and torch.isfinite(out['probabilities']).all()
    out['loss'].backward()
    assert z.grad.count_nonzero()==0


@pytest.mark.parametrize('weight',[-1.,float('inf'),float('nan')])
def test_invalid_soft_iou_weight_fails(weight):
    with pytest.raises(ValueError):
        decision_losses(torch.zeros(1,2),torch.ones(1,2,dtype=torch.bool),torch.tensor([CHOICE]),
                        torch.tensor([[1.,0.]]),soft_iou_weight=weight)
