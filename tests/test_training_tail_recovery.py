"""Metadata-only tail selection retains the original distributed update schedule."""
import importlib.util
from pathlib import Path

import pytest
import torch

from timejev.packing import PackedBatchSampler


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('tail_training', ROOT/'scripts/train.py')
training = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training)


def make_sampler(rank):
    return PackedBatchSampler([1]*5931, 147456, rank, 8, 20260924, 32,
                              packs=[[i] for i in range(5931)])


def test_last_two_real_updates_keep_32_then_11_packs_and_sparse_ranks():
    seen = []
    for rank in range(8):
        base = make_sampler(rank)
        view = training.PackedUpdateView(base, 185)
        original = list(base)
        selected = list(view)
        assert selected == [batch for batch in original if batch[0][1] >= 185]
        assert len(view) == 6
        assert [batch[0][1] for batch in selected] == [185]*4+[186]*2
        assert [batch[0][3] for batch in selected] == [4]*4+[2]*2
        assert [batch[0][5] for batch in selected] == [32]*4+[11]*2
        assert (selected[-1][0][0] is None) == (rank >= 3)
        seen.extend(item[0] for batch in selected for item in batch if item[0] is not None)
    assert len(seen) == len(set(seen)) == 43


def test_mid_epoch_resume_starts_only_at_saved_next_update():
    for rank in range(8):
        sampler = make_sampler(rank)
        view = training.PackedUpdateView(sampler, 186)
        assert len(view) == 2
        assert list(view) == list(sampler)[-2:]
        view.epoch = 1
        assert sampler.epoch == 1
        assert list(view) == list(sampler)[-2:]


@pytest.mark.parametrize('index', [0, -1, 187])
def test_invalid_update_cannot_silently_change_training_data(index):
    with pytest.raises(ValueError):
        training.PackedUpdateView(make_sampler(0), index)


def test_rng_capture_restore_includes_python_numpy_torch_and_cuda(monkeypatch):
    import random
    import numpy as np
    cuda = {'state': torch.tensor([1, 3, 7], dtype=torch.uint8)}
    monkeypatch.setattr(torch.cuda, 'get_rng_state', lambda: cuda['state'].clone())
    monkeypatch.setattr(torch.cuda, 'set_rng_state', lambda value: cuda.update(state=value.clone()))
    state = training.capture_rng()
    expected = (random.random(), float(np.random.random()), torch.rand(5))
    cuda['state'].zero_()
    training.restore_rng(state)
    actual = (random.random(), float(np.random.random()), torch.rand(5))
    assert expected[:2] == actual[:2]
    torch.testing.assert_close(expected[2], actual[2], atol=0, rtol=0)
    torch.testing.assert_close(cuda['state'], torch.tensor([1, 3, 7], dtype=torch.uint8))
