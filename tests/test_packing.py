from pathlib import Path
import os
import collections
import math
import numpy as np
import pytest
import torch
from timejev.packing import PackedBatchSampler, best_fit, PackedCollator
from timejev.vision import load_processor, resolution_for_budget, prepare_video
from timejev.features import cached_inputs
from timejev.adaptive_features import fast_cached_inputs


@pytest.mark.parametrize('global_batch',[32,64,128])
def test_global_batch_and_balanced_collective_counts(global_batch):
    from timejev.packing import build_xtuner_packs
    lengths=[127,129,4600,1234,17000,300,9500]*200
    packs=build_xtuner_packs(lengths,20032,314)
    samplers=[PackedBatchSampler(lengths,20032,r,8,314,global_batch,packs=packs) for r in range(8)]
    assert all(s.plan()==samplers[0].plan() for s in samplers)
    by_update=collections.defaultdict(list)
    seen_packs=[]
    for entry in samplers[0].plan():
        for pack_id,pack in zip(entry['rank_pack_ids'],entry['rank_ids']):
            assert sum(lengths[i] for i in pack)<=20032
            by_update[entry['update']].extend(pack)
            if pack_id is not None:seen_packs.append(pack_id)
    assert len(by_update)==math.ceil(len(packs)/global_batch)
    assert sorted(seen_packs)==list(range(len(packs)))
    for update,ids in by_update.items():
        expected=next(p for p in samplers[0].plan() if p['update']==update)
        assert len(expected['global_pack_ids'])==min(global_batch,len(packs)-(update-1)*global_batch)
        assert collections.Counter(ids)==collections.Counter(expected['global_ids'])
    counts=collections.Counter(i for ids in by_update.values() for i in ids)
    assert set(counts)==set(range(len(lengths))) and all(n==1 for n in counts.values())
    # Query-mean BCE is independent of how many original queries fit each pack.
    losses=np.arange(len(lengths),dtype=np.float64)/13
    for update,ids in by_update.items():
        gradient=0.
        for p in samplers[0].plan():
            if p['update']!=update:continue
            for pack in p['rank_ids']:
                if pack:
                    accumulation=global_batch//8
                    scaled=losses[pack].mean()*len(pack)*8*accumulation/len(ids)
                    gradient+=scaled/accumulation/8
        assert gradient==pytest.approx(losses[ids].mean())


def test_no_splitting_or_silent_truncation():
    with pytest.raises(ValueError):best_fit([0],[65],64)
    assert best_fit([0,1,2],[60,30,30],64)==[[0],[1,2]]


@pytest.fixture(scope='module')
def processor():
    model = os.environ.get('VIDEOJEV_TEST_MODEL', 'models/Qwen3.5-4B')
    if not Path(model).is_dir():
        pytest.skip('Set VIDEOJEV_TEST_MODEL to a local Qwen3.5-4B model')
    return load_processor(model)


@pytest.mark.parametrize('frames,grid',[(3,[2,4,6]),(2,[1,8,4]),(17,[9,6,10])])
@pytest.mark.parametrize('placement',['both','last'])
def test_fast_token_expansion_is_official(processor,frames,grid,placement):
    record=dict(grid_thw=grid,frames=frames,fps=2.,tokens=math.prod(grid)//4)
    for query in ['A person opens the door.', '先拿杯子，然后转身。']:
        reference=cached_inputs(processor,record,query,placement)
        fast=fast_cached_inputs(processor,record,query,placement)
        for key in reference:
            torch.testing.assert_close(fast[key],reference[key],rtol=0,atol=0)
        assert fast_cached_inputs(processor,record,query,placement,lengths_only=True)==reference['input_ids'].shape[1]


@pytest.mark.parametrize('frames,h,w,budget',[(984,1080,1920,52428),(981,1920,1080,52428),(600,640,640,52428),(4,1080,1920,64)])
def test_native_resolution_budget(processor,frames,h,w,budget):
    r=resolution_for_budget(processor,frames,h,w,budget)
    assert r['visual_tokens']<=budget
    assert r['max_frame_pixels']<=409600
    hh,ww=r['processed_hw']
    assert r['visual_tokens']==((frames+1)//2)*(hh//32)*(ww//32)
    # The cap is the largest integer cap accepted by the native resize function.
    if r['max_frame_pixels']<409600:
        from transformers.models.qwen3_vl.video_processing_qwen3_vl import smart_resize
        hh,ww=smart_resize(frames,h,w,min_pixels=processor.video_processor.size['shortest_edge'],
                          max_pixels=frames*(r['max_frame_pixels']+1))
        assert ((frames+1)//2)*(hh//32)*(ww//32)>budget


def test_resize_keeps_timeline_and_frames(processor):
    n=4
    sample=dict(video=np.zeros((n,360,640,3),dtype=np.uint8),times=np.arange(n)/2,
                decoded_times=np.arange(n)/2,fps=2.,media_duration=2.,sampled_duration=2.,ignored_duplicate_pts=0)
    inputs,audit=prepare_video(processor,sample,'The door opens.',visual_token_budget=64)
    assert audit['sampled_frames']==n and audit['temporal_groups']==2
    assert audit['visual_tokens']<=64
    assert inputs['video_grid_thw'][0,0]==2


def make_items(processor,hidden=128):
    items=[]
    for i,(frames,grid) in enumerate([(4,[2,4,6]),(7,[4,6,4]),(2,[1,4,4])]):
        record=dict(frames=frames,grid_thw=grid,fps=2.,tokens=math.prod(grid)//4)
        inp=fast_cached_inputs(processor,record,['Open the door.','A person sits down.','Close the window.'][i])
        inp['visual_features']=torch.randn(record['tokens'],hidden,dtype=torch.bfloat16)
        items.append(dict(inputs=inp,times=np.arange(grid[0]),labels=torch.arange(grid[0]).float()%2,
                          row={'id':str(i)},audit={'feature_cache_id':'test'}))
    return items


def test_packed_boundaries_and_readout_offsets(processor):
    items=make_items(processor)
    p=PackedCollator(processor.tokenizer,4096)(items)
    offset=0
    for i,x in enumerate(items):
        n=x['inputs']['input_ids'].shape[1]
        torch.testing.assert_close(p['input_ids'][:,offset:offset+n],x['inputs']['input_ids'])
        assert p['cu_seq_lens_q'][i]==offset
        assert p['seq_idx'][0,offset:offset+n].eq(i).all()
        assert offset<=p['query_indices'][i]<offset+n
        assert ((p['pair_indices'][i,p['score_mask'][i]]>=offset)&(p['pair_indices'][i,p['score_mask'][i]]<offset+n)).all()
        offset+=n
    assert p['sample_count']==3
    assert p['attention_mask'] is None
    assert p['cu_seq_lens_q'][-1]==p['input_ids'].shape[1]
