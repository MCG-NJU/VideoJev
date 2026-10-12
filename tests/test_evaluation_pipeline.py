"""Exercise all seven exports, completion reuse and guarded cleanup without GPUs."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import evaluate, pipeline
from timejev.data import read_jsonl

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('evaluation_folder', ['evaluation', 'evaluation_tail_recovery_v1/evaluation'])
def test_seven_benchmark_exports_retry_and_cleanup(tmp_path,monkeypatch,evaluation_folder):
    if not (Path(__file__).resolve().parents[1]/'data/ready.json').exists():
        pytest.skip('Required prepared data is not bundled with the source release')
    c=json.loads((ROOT/'configs/joint.json').read_text())
    run=tmp_path/'run';checkpoint=run/'joint/model.pt';checkpoint.parent.mkdir(parents=True)
    torch.save(dict(config=c,checkpoint_id='test-identity',step=1,model={}),checkpoint)
    calls=[]
    class Model:
        last_video_cache_stats={}
        def load_checkpoint(self,value):calls.append('load')
        def to(self,device):return self
        def eval(self):return self
    class Dataset:
        def __init__(self,manifest,*args,**kwargs):self.rows=read_jsonl(manifest)
    monkeypatch.setattr(evaluate,'setup',lambda seed:(0,1,'cpu'))
    monkeypatch.setattr(evaluate,'from_config',lambda *args,**kwargs:Model())
    monkeypatch.setattr(evaluate,'register_training_caches',lambda *args:None)
    monkeypatch.setattr(evaluate,'validate_evaluation',lambda *args:{})
    monkeypatch.setattr(evaluate,'validate_cache',lambda *args,**kwargs:{})
    monkeypatch.setattr(evaluate,'load_processor',lambda *args:SimpleNamespace(tokenizer=None))
    monkeypatch.setattr(evaluate,'data_module',lambda config:SimpleNamespace(DecisionDataset=Dataset,
        PackedDecisionCollator=lambda *args:None,load_lengths=lambda *args:[1]*len(args[-1])))
    monkeypatch.setattr(evaluate,'make_loader',lambda dataset,*args,**kwargs:dataset)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    def qa(model,data,device):
        calls.append('qa')
        return [dict(id=r['id'],task='choice',answer_index=r['answer_index'],options=r['options'],
            probabilities=[float(i==r['answer_index']) for i in range(len(r['options']))],
            logits=[float(i==r['answer_index']) for i in range(len(r['options']))]) for r in data.rows]
    def vtg(model,data,device):
        calls.append('vtg')
        return [dict(id=r['id'],times=[r['duration']/2],logits=[1.],bounds=[[0.,r['duration']]],
            sampled_duration=r['duration'],audit={}) for r in data.rows]
    monkeypatch.setattr(evaluate,'predict_decisions',qa)
    monkeypatch.setattr(evaluate,'predict',vtg)
    monkeypatch.setattr(evaluate, 'predict_dataset',
        lambda model, processor, dataset, indices, config, device:
            (qa if config.get('qa_task') == 'choice' else vtg)(model, dataset, device))
    # Decoder selection itself is tested against continuous/final policy fixtures.
    monkeypatch.setattr(evaluate,'select_decoder',lambda *args:dict(threshold=.5,smooth=1,postprocess='round_repair_clamp_v1'))
    target=run/evaluation_folder
    monkeypatch.setattr('sys.argv',['evaluate','--checkpoint',str(checkpoint),'--out',str(target)])
    evaluate.main()
    assert calls==['load','vtg','vtg','vtg','qa','qa','qa','qa']
    assert set(json.loads((target/'summary.json').read_text()))==pipeline.BENCHMARKS
    state=pipeline.verify_completion(target)
    assert state['questions']==17164
    calls.clear();evaluate.main();assert not calls
    for stage in ('head_only','joint'):
        d=run/stage;d.mkdir(exist_ok=True)
        (d/'completion.json').write_text(json.dumps(dict(complete=True,steps=1)))
        (d/'training_config.json').write_text(json.dumps(dict(world_size=64,total_steps=1,unique_training_queries=10)))
        for rank in range(64):
            (d/f'train_rank{rank}.jsonl').write_text(json.dumps(dict(step=1,global_samples=10,vtg_queries=5,qa_queries=5))+'\n')
        (d/'safety_deepspeed').mkdir()
    (run/'head_only/head.pt').write_text('retained head')
    pipeline.cleanup(run, target if evaluation_folder != 'evaluation' else None)
    assert (run/'completion.json').exists() and (run/'RESULTS.md').exists()
    assert not (run/'joint/safety_deepspeed').exists()
    assert checkpoint.exists() and (run/'head_only/head.pt').exists()
    (target/'lvbench/metrics.json').write_text('{}')
    with pytest.raises(ValueError,match='artifact changed'):
        evaluate.main()
