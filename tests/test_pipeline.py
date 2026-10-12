import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('pipeline', ROOT/'scripts/pipeline.py')
pipeline=importlib.util.module_from_spec(spec);spec.loader.exec_module(pipeline)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def test_snapshot_pins_both_stages_and_refuses_changed_snapshot(tmp_path):
    project=tmp_path/'project';run=tmp_path/'run'
    for name in ('timejev','scripts','configs','videojev_hf','.deps'):(project/name).mkdir(parents=True)
    (project/'videojev_hf/modeling_videojev.py').write_text('# bundled inference runtime')
    for name in ('sitecustomize.py','requirements-overlay.txt'):(project/name).write_text('')
    write(project/'data/ready.json',{'complete':True})
    for stage in ('head_only','joint'):write(project/f'configs/{stage}.json',{'stage':stage})
    source=pipeline.snapshot(project,run)
    assert (source/'videojev_hf/modeling_videojev.py').read_text() == '# bundled inference runtime'
    assert json.loads((source/'configs/joint.json').read_text())['deepspeed']==str(source/'configs/deepspeed_zero2.json')
    write(project/'configs/head_only.json',{'changed':True})
    assert pipeline.snapshot(project,run)==source
    write(source/'configs/head_only.json',{'changed':True})
    with pytest.raises(ValueError,match='snapshot changed'):pipeline.snapshot(project,run)


def test_resume_uses_complete_inventory_and_skip_needs_artifact(tmp_path):
    assert pipeline.action(tmp_path,'head_only')=='start'
    d=tmp_path/'head_only'
    write(d/'training_config.json',{})
    with pytest.raises(ValueError,match='No complete'):pipeline.action(tmp_path,'head_only')
    tag=d/'safety_deepspeed/step000100'
    write(tag/'weights.pt',{})
    write(tag/'complete_manifest.json',dict(complete=True,world_size=64,files={'weights.pt':2}))
    assert pipeline.action(tmp_path,'head_only')=='resume'
    write(d/'completion.json',{'complete':True})
    assert pipeline.action(tmp_path,'head_only')=='resume'
    (d/'head.pt').write_text('head')
    assert pipeline.action(tmp_path,'head_only')=='skip'


def test_missing_benchmark_never_deletes_resume_states(tmp_path):
    for stage in ('head_only','joint'):
        (tmp_path/stage/'safety_deepspeed').mkdir(parents=True)
    write(tmp_path/'evaluation/completion.json',dict(complete=True,files={'summary.json':'wrong'},benchmarks=[]))
    with pytest.raises(ValueError):pipeline.cleanup(tmp_path)
    assert (tmp_path/'head_only/safety_deepspeed').exists()
    assert (tmp_path/'joint/safety_deepspeed').exists()


@pytest.mark.parametrize('rank',range(8))
def test_all_eight_launch_ranks_use_64_gpus_and_both_stages(rank,tmp_path):
    env=dict(os.environ,NODE_RANK=str(rank),MASTER_ADDR='192.0.2.10',WORK_DIR=str(tmp_path/'unused'))
    for name in ('LOCAL_RANK','NNODES','NODE_COUNT','SLURM_JOB_NUM_NODES','SLURM_NTASKS_PER_NODE'):
        env.pop(name,None)
    result=subprocess.run(['bash',str(ROOT/'scripts/train_eval_8nodes.sh'),'--dry-run'],env=env,text=True,capture_output=True)
    assert result.returncode==0,result.stderr
    assert result.stdout.count('--nnodes=8 --nproc-per-node=8')==3
    assert f'--node-rank={rank}' in result.stdout
    assert '--init-head' in result.stdout and 'scripts/evaluate.py' in result.stdout
    assert not (tmp_path/'unused').exists()
