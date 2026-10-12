"""Scale invariance, mixed decisions, checkpoint protocol and isolated recipes."""
import copy
import json
from pathlib import Path

import pytest
import torch

from timejev.model import PointerHead, TimeJev
from timejev.decision_model import typed_outputs
from timejev.vision import VISUAL_PROTOCOL


def test_cosine_removes_projection_scale_without_new_parameters_or_rng_changes():
    torch.manual_seed(20260924)
    original = PointerHead(32, 16)
    rng = torch.random.get_rng_state().clone()
    torch.manual_seed(20260924)
    head = PointerHead(32, 16, similarity='cosine', logit_scale=10.)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert list(head.state_dict()) == list(original.state_dict())
    for name, value in original.state_dict().items():
        torch.testing.assert_close(head.state_dict()[name], value, atol=0, rtol=0)
    pairs, query = torch.randn(4, 7, 32), torch.randn(4, 32)
    before = head(pairs, query)
    with torch.no_grad():
        head.q.weight.mul_(1000);head.q.bias.mul_(1000)
        head.k.weight.mul_(.01);head.k.bias.mul_(.01)
    torch.testing.assert_close(head(pairs, query), before, atol=3e-6, rtol=3e-6)
    assert (before-head.bias).abs().max() <= 10.00001
    assert not any('scale' in name for name, _ in head.named_parameters())


def test_cosine_grounding_and_qa_have_finite_gradients_and_ignore_padding():
    torch.manual_seed(35)
    head = PointerHead(8, 4, similarity='cosine')
    pairs = torch.randn(3, 4, 8, requires_grad=True)
    query = torch.randn(3, 8, requires_grad=True)
    mask = torch.tensor([[1,1,0,0],[1,1,1,0],[0,0,0,0]],dtype=torch.bool)
    labels = torch.tensor([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,0.,0.]])
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3)
    out = typed_outputs(head(pairs,query),mask,torch.tensor([0,1,0]),labels)
    torch.testing.assert_close(out['probabilities'][1].sum(),torch.tensor(1.))
    assert torch.isfinite(out['loss']) and out['probabilities'][~mask].count_nonzero()==0
    out['loss'].backward()
    assert pairs.grad[~mask].count_nonzero()==0 and query.grad[2].count_nonzero()==0
    for p in head.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.count_nonzero()>0
    opt.step()
    restored = PointerHead(8,4,similarity='cosine')
    restored.load_state_dict(head.state_dict(),strict=True)
    torch.testing.assert_close(restored(pairs,query),head(pairs,query),atol=0,rtol=0)


def test_zero_projections_are_finite_and_keep_bias():
    head=PointerHead(8,4,similarity='cosine')
    with torch.no_grad():
        for p in head.parameters():p.zero_()
        head.bias.fill_(.3)
    z=head(torch.zeros(2,3,8),torch.zeros(2,8))
    torch.testing.assert_close(z,torch.full_like(z,.3),atol=0,rtol=0)
    z.sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters())


def test_confirmed_stages_reject_changed_loss_or_batch_recipe():
    from scripts.train import validate_recipe
    root = Path(__file__).resolve().parents[1]
    for stage in ('head_only', 'joint'):
        config = json.loads((root / f'configs/{stage}.json').read_text())
        validate_recipe(config)
        for change in ({'global_batch_size': 32}, {'soft_iou_weight': 2. if stage == 'head_only' else 0.}):
            with pytest.raises(ValueError):
                validate_recipe(dict(config, **change))


@pytest.mark.parametrize('stored,current,stored_scale,current_scale,allowed',[
    ('dot_product','cosine',10.,10.,False),('cosine','dot_product',10.,10.,False),
    ('cosine','cosine',10.,20.,False),('cosine','cosine',10.,10.,True)])
def test_checkpoint_rejects_wrong_scoring_before_weights(stored,current,stored_scale,current_scale,allowed):
    model=TimeJev.__new__(TimeJev);torch.nn.Module.__init__(model)
    model.attention_mode='causal';model.query_placement='both';model.feature_cache_id=None
    model.head_similarity=current;model.head_logit_scale=current_scale
    checkpoint=dict(visual_protocol=VISUAL_PROTOCOL,model={},config=dict(
        head_similarity=stored,head_logit_scale=stored_scale))
    loaded=[]
    model.load_state_dict=lambda *args,**kwargs:loaded.append(True)
    if allowed:
        model.load_checkpoint(checkpoint);assert loaded
    else:
        with pytest.raises(ValueError,match='similarity or scale'):model.load_checkpoint(checkpoint)
        assert not loaded
