"""Eleven complete cached-feature evaluations using same-video query chunks."""
import argparse
from timejev.hf_checkpoint import load_training_checkpoint, checkpoint_fingerprint, checkpoint_stat
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
import torch.distributed as dist
from safetensors.torch import load_file

from scripts.evaluate import barrier, cached_completion, mark_complete, read, score_qa
from scripts.evaluate_cached_contexts import setup, broadcast, load_cached_model
from timejev.cached_context_features import validate_checkpoint_vision
from timejev.context_eval import digest_json
from timejev.data import read_jsonl
from timejev.decision_inputs import load_processor
from timejev.engine import atomic_save, save_json, sha256
from timejev.multiturn_evaluation import PROTOCOL, predict_video
from timejev.rgb_media import timeline
from timejev.temporal import decode, metrics

QA = ('lvbench','videomme','longvideobench','mlvu')
VTG = ('charades','activitynet','qvhighlights')
COUNTS = dict(charades=3363,activitynet=4500,qvhighlights=1541,
              lvbench=1549,videomme=2700,longvideobench=1337,mlvu=2174)


def target_for(output, scope):
    context, name = scope.split('/')
    return output/'grounding'/name if name in VTG else output/f'qa/context_{context}'/name


def assignments(jobs, world):
    loads, result = [0]*world, [[] for _ in range(world)]
    for job in sorted(jobs,key=lambda j:-(j['tokens'] or j['record']['tokens'])):
        rank = min(range(world),key=lambda r:loads[r])
        result[rank].append(job)
        loads[rank] += job['tokens'] or job['record']['tokens']
    return result


def validate_plan(plan, path):
    expected = {f'{c}/{n}' for n in COUNTS for c in ((147456,) if n in VTG else (196608,262144))}
    if plan['protocol'] != PROTOCOL or set(plan['scopes']) != expected:
        raise ValueError('Expected all eleven complete multi-query benchmarks')
    for name, digest in plan['files'].items():
        if sha256(path.parent/name) != digest:
            raise ValueError(f'Frozen multi-query reference changed: {name}')
    cache = Path(plan['cache'])
    for name,key in [('model.json','cache_model_sha256'),('completion.json','cache_completion_sha256')]:
        if sha256(cache/name) != plan[key]:
            raise ValueError(f'Frozen evaluation feature cache changed: {name}')
    for scope,spec in plan['scopes'].items():
        rows = read_jsonl(path.parent/spec['manifest'])
        jobs = [j for j in plan['jobs'] if j['scope']==scope]
        indices = [i for j in jobs for i in j['query_indices']]
        if (len(rows)!=COUNTS[spec['name']] or len({r['id'] for r in rows})!=len(rows)
                or sorted(indices)!=list(range(len(rows)))
                or len({j['key'] for j in jobs})!=len(jobs)
                or len({j['video_key'] for j in jobs})!=len(jobs)):
            raise ValueError(f'Missing/duplicate evaluation questions: {scope}')
        for job in jobs:
            if (job['context']!=spec['context'] or job['task']!=spec['task']
                    or job['query_indices']!=sorted(job['query_indices'])
                    or any(rows[i]['video_key']!=job['video_key'] for i in job['query_indices'])):
                raise ValueError(f'Query order or video identity changed: {scope}')


def read_job(path, identity, job, rows):
    if not path.exists():
        return None
    value = read(path)
    payload = value['payload']
    if (value['identity']!=identity or value['job_sha256']!=digest_json(job)
            or value['payload_sha256']!=digest_json(payload)
            or [p['id'] for p in payload['predictions']] != [rows[i]['id'] for i in job['query_indices']]):
        raise ValueError(f'Completed multi-query output changed: {path}')
    return payload


def read_features(job):
    feature = job['feature']
    path = Path(feature['path'])
    if path.stat().st_size != feature['bytes'] or sha256(path) != feature['sha256']:
        raise ValueError(f'Cached feature checksum changed: {path}')
    value = load_file(str(path))
    features = value['features']
    if (features.dtype!=torch.bfloat16 or list(features.shape)!=[feature['tokens'],2560]
            or value['video_grid_thw'].tolist()!=[feature['grid_thw']]):
        raise ValueError(f'Cached feature shape/dtype/grid changed: {path}')
    return features


def finish_scope(target, spec, rows, payloads, identity, decoder):
    predictions = [p for payload in payloads for p in payload['predictions']]
    by_id = {p['id']:p for p in predictions}
    if len(predictions)!=len(rows) or len(by_id)!=len(rows) or set(by_id)!={r['id'] for r in rows}:
        raise ValueError('Missing/duplicate completed multi-query predictions')
    ordered = [by_id[r['id']] for r in rows]
    files = []
    if spec['name'] in QA:
        result = score_qa(spec['name'],ordered,rows,spec['reference'],target)
        files += [target/'answers.json',target/'predictions.jsonl']
        if spec['name']=='videomme': files.append(target/'official_results.json')
    else:
        spans = [decode(p['logits'],p['times'],p['sampled_duration'],decoder['threshold'],decoder['smooth'],
                 bounds=p['bounds'],postprocess=decoder['postprocess']) for p in ordered]
        result = metrics(spans,rows)
        result.update(decoder=decoder,decoder_source='evaluation plan decoder')
        (target/'predictions.jsonl').write_text(''.join(json.dumps(dict(p,prediction=s),ensure_ascii=False)+'\n'
                                               for p,s in zip(ordered,spans)))
        files.append(target/'predictions.jsonl')
    result.update(n=len(rows),expected_questions=len(rows),evaluated_questions=len(rows),
        full_coverage=True,skipped_questions=0,protocol=PROTOCOL,context=spec['context'],
        forward_calls=sum(p['forward_calls'] for p in payloads),videos=len(payloads),
        checkpoint_id=identity['checkpoint_id'])
    save_json(result,target/'metrics.json')
    atomic_save(dict(scores=ordered,checkpoint_id=identity['checkpoint_id']),target/'scores.pt')
    mark_complete(target,identity,[*files,target/'metrics.json',target/'scores.pt'])
    return result


def finish_evaluation(output, summary, identity):
    grounding = {n:summary[f'147456/{n}'] for n in VTG}
    save_json(grounding,output/'grounding/vtg_summary.json')
    mark_complete(output/'grounding',identity,[output/'grounding/vtg_summary.json',
        *(output/'grounding'/n/'completion.json' for n in VTG)])
    comparison = {}
    for context in (196608,262144):
        target = output/f'qa/context_{context}'
        comparison[str(context)] = {n:summary[f'{context}/{n}'] for n in QA}
        save_json(comparison[str(context)],target/'summary.json')
        mark_complete(target,identity,[target/'summary.json',*(target/n/'completion.json' for n in QA)])
    from timejev.qa_eval_settings import write_context_results
    selected_files = write_context_results(comparison, output/'qa')
    save_json(comparison,output/'qa/summary.json')
    mark_complete(output/'qa',identity,[output/'qa/summary.json', *selected_files,
        *(output/f'qa/context_{c}/completion.json' for c in (196608,262144))])
    mark_complete(output,identity,[output/'grounding/completion.json',output/'qa/completion.json'],
                  questions=sum(v['n'] for v in summary.values()),full_coverage=True)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--plan',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--expected-world-size',type=int)
    p.add_argument('--expected-checkpoint-id')
    p.add_argument('--expected-checkpoint-sha256')
    args = p.parse_args()
    checkpoint_path, plan_path, output = Path(args.checkpoint).resolve(),Path(args.plan).resolve(),Path(args.out).resolve()
    plan = read(plan_path)
    before = checkpoint_stat(checkpoint_path)
    checkpoint = load_training_checkpoint(checkpoint_path)
    config = checkpoint['config']
    if config['base']!=plan['base'] or config['training_stage']!='joint' or checkpoint['step']<=0:
        raise ValueError('Expected a completed Video-Jev-4B joint checkpoint')
    rank,world,device = setup(config['seed'])
    if args.expected_world_size is not None and world != args.expected_world_size:
        raise ValueError(f'Expected {args.expected_world_size} evaluation workers, got {world}')
    if args.expected_checkpoint_id and checkpoint['checkpoint_id'] != args.expected_checkpoint_id:
        raise ValueError('Checkpoint ID differs from the frozen reevaluation task')
    if rank==0:
        validate_plan(plan,plan_path)
        validate_checkpoint_vision(checkpoint,plan['cache'])
    barrier(world)
    root = Path(__file__).resolve().parents[1]
    code_hash = digest_json({str(f.relative_to(root)):sha256(f) for folder in ('timejev','scripts')
                            for f in sorted((root/folder).glob('*.py'))})
    identity = dict(checkpoint_id=checkpoint['checkpoint_id'],
        checkpoint_sha256=broadcast(checkpoint_fingerprint(checkpoint_path) if rank==0 else None,world),
        code_sha256=code_hash,plan_sha256=sha256(plan_path),evaluation_protocol=PROTOCOL)
    if args.expected_checkpoint_sha256 and identity['checkpoint_sha256'] != args.expected_checkpoint_sha256:
        raise ValueError('Checkpoint weights differ from the frozen reevaluation task')
    model,processor,summary = None,None,{}
    for scope,spec in plan['scopes'].items():
        target = target_for(output,scope)
        target.mkdir(parents=True,exist_ok=True)
        scope_identity = dict(identity,scope=scope)
        done = broadcast(cached_completion(target,scope_identity) if rank==0 else None,world)
        if not done:
            rows = read_jsonl(plan_path.parent/spec['manifest'])
            jobs = [j for j in plan['jobs'] if j['scope']==scope]
            (target/'videos').mkdir(exist_ok=True)
            for job in assignments(jobs,world)[rank]:
                path = target/'videos'/f"{job['key']}.json"
                if read_job(path,scope_identity,job,rows) is not None:
                    continue
                if model is None:
                    model = load_cached_model(checkpoint,device)
                    model.feature_cache_ids.update(j['record']['cache_id'] for j in plan['jobs'])
                    processor = load_processor(config['base'])
                    del checkpoint
                start = time.monotonic()
                features = read_features(job)
                record = timeline(dict(job['record'])) if job['task']=='choice' else job['record']
                payload = predict_video(model,processor,features,record,
                    [rows[i] for i in job['query_indices']],job['context'],device)
                payload['seconds'] = time.monotonic()-start
                save_json(dict(identity=scope_identity,job_sha256=digest_json(job),payload=payload,
                    payload_sha256=digest_json(payload)),path)
                print(json.dumps(dict(event='video_complete',scope=scope,video=job['video_key'],
                    questions=len(payload['predictions']),forward_calls=payload['forward_calls'])),flush=True)
                del features,payload
            barrier(world)
            if rank==0:
                payloads = [read_job(target/'videos'/f"{j['key']}.json",scope_identity,j,rows) for j in jobs]
                if any(v is None for v in payloads): raise ValueError('Incomplete multi-query video coverage')
                resolved = dict(spec,reference={k:str(plan_path.parent/v) for k,v in spec['reference'].items()})
                finish_scope(target,resolved,rows,payloads,scope_identity,plan['decoder'])
            barrier(world)
        if rank==0: summary[scope] = read(target/'metrics.json')
    if rank==0:
        after = checkpoint_stat(checkpoint_path)
        if before != after:
            raise ValueError('Checkpoint changed during evaluation')
        finish_evaluation(output,summary,identity)
    barrier(world)
    if dist.is_initialized(): dist.destroy_process_group()


if __name__=='__main__':
    main()
