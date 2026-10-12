"""Evaluate 4B checkpoints with the frozen 256K offline ViT caches.

Adapted from the completed context-evaluation snapshot; scoring, text inputs,
whole-video query grouping and verified output/resume contracts are used.
"""
from timejev.config import read_config
import argparse
from timejev.hf_checkpoint import load_training_checkpoint, checkpoint_fingerprint, checkpoint_stat
from datetime import timedelta
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

from scripts.evaluate import barrier, cached_completion, mark_complete, read, score_qa
from timejev.context_eval import (CONTEXTS, COUNTS, PROTOCOL, assignments, check_source,
    digest_json, question_item, validate_spec, visual_budget)
from timejev.data import read_jsonl, to_device
from timejev.decision_inputs import FORMULATION, load_processor
from timejev.decision_model import from_config
from timejev.engine import atomic_save, make_loader, save_json, sha256
from timejev.multiturn_data import MultiturnCollator
from timejev.cached_context_features import (CachedEvaluationVideos, single_video_collate,
    validate_feature_cache, validate_checkpoint_vision, CACHE_PROTOCOL)


def setup(seed):
    rank, world, local = (int(os.environ.get(k, d)) for k, d in
                          (('RANK', 0), ('WORLD_SIZE', 1), ('LOCAL_RANK', 0)))
    device = torch.device('cuda', local)
    torch.cuda.set_device(device)
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    if world > 1:
        # Whole-video shards differ in duration, especially LVBench (103 videos).
        dist.init_process_group('nccl', device_id=device, timeout=timedelta(hours=6))
    return rank, world, device


def broadcast(value, world):
    envelope = [value]
    if world > 1:
        dist.broadcast_object_list(envelope)
    return envelope[0]


def load_cached_model(checkpoint, device):
    config = checkpoint['config']
    if config.get('input_mode') == 'rgb_dynamic':
        raise ValueError('This experiment evaluates the original completed cached-feature model')
    model = from_config(config, checkpointing=False)
    model.load_checkpoint(checkpoint)  # Preserve all architecture/metadata/weight checks.
    model.feature_cache_ids.add(PROTOCOL)
    return model.to(device).eval()


def video_path(target, video):
    return target/'videos'/(digest_json(video['video_key'])+'.json')


def read_video(path, identity, video):
    if not path.exists():
        return None
    value = read(path)
    if (value['identity'] != identity or value['video_key'] != video['video_key']
            or value['payload_sha256'] != digest_json(value['payload'])
            or [p['id'] for p in value['payload']['predictions']] != [q['id'] for q in video['questions']]):
        raise ValueError(f'Completed video output changed: {path}')
    return value['payload']


@torch.inference_mode()
def predict_videos(model, loader, spec, rows, config, device, processor, *, on_video=None):
    from timejev.multiturn_evaluation import predict_video
    from timejev.rgb_media import timeline
    context = config['max_sequence_length']
    results, statistics = [], []
    for payload in loader:
        video = spec['videos'][payload['index']]
        began = time.monotonic()
        features = payload['visual_features'].to(device)
        members = [rows[q['query_index']] for q in video['questions']]
        if [r['id'] for r in members] != [q['id'] for q in video['questions']]:
            raise ValueError('Question differs from the immutable RGB plan')
        record = timeline(dict(video['record'], **video['geometry'], cache_id=PROTOCOL))
        prediction = predict_video(model, processor, features, record, members, context, device)
        stats = dict(video_key=video['video_key'], questions=len(members), hits=0, misses=1,
                     queries=prediction['forward_calls'], forward_calls=prediction['forward_calls'],
                     seconds=time.monotonic()-began)
        value = dict(predictions=prediction['predictions'], cache=stats)
        if on_video:
            on_video(video, value)
        else:
            results.extend(value['predictions'])
        statistics.append(stats)
        print(json.dumps(dict(event='video_complete', context=context, video=video['video_key'],
            questions=len(members), forward_calls=prediction['forward_calls'])), flush=True)
        del features, payload
    return results, statistics


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--plans', required=True)
    p.add_argument('--feature-cache', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--workers', type=int, default=1)
    a = p.parse_args()
    if a.workers < 0:
        p.error('--workers must be nonnegative')
    checkpoint_path = Path(a.checkpoint).resolve(strict=True)
    before = checkpoint_stat(checkpoint_path)
    checkpoint = load_training_checkpoint(checkpoint_path)
    config, checkpoint_id = dict(checkpoint['config']), checkpoint['checkpoint_id']
    if config.get('training_stage') != 'joint' or checkpoint['step'] <= 0:
        raise ValueError('Expected a completed joint-stage checkpoint')
    rank, world, device = setup(config['seed'])
    output, plans = Path(a.out), Path(a.plans)
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    code_hash = digest_json({str(p.relative_to(root)): sha256(p)
        for folder in ('timejev', 'scripts') for p in sorted((root/folder).glob('*.py'))})
    checkpoint_hash = broadcast(checkpoint_fingerprint(checkpoint_path) if rank == 0 else None, world)
    cache = Path(a.feature_cache).resolve(strict=True)
    cache_specs = validate_feature_cache(cache, plans, config['base'])
    if rank == 0:
        validate_checkpoint_vision(checkpoint, cache)
    barrier(world)
    common = dict(checkpoint_id=checkpoint_id, checkpoint_sha256=checkpoint_hash,
                  code_sha256=code_hash, input_mode=CACHE_PROTOCOL,
                  cache_model_sha256=sha256(cache/'model.json'))
    model, processor, qa_results = None, None, {}
    for context in CONTEXTS:
        plan_path = plans/f'context_{context}.json'
        plan = read(plan_path)
        if (plan['context'] != context or plan['protocol'] != PROTOCOL or plan['base'] != config['base']
                or plan['visual_budget'] != visual_budget(context) or set(plan['benchmarks']) != set(COUNTS)):
            raise ValueError('Unexpected RGB plan/context/base identity')
        for name, digest in plan['identity']['code'].items():
            if sha256(root/name) != digest:
                raise ValueError(f'Planning code differs from prepared RGB inputs: {name}')
        directory = output/f'context_{context}'
        directory.mkdir(exist_ok=True)
        summary = {}
        for name in COUNTS:
            spec = plan['benchmarks'][name]
            rows = read_jsonl(spec['manifest'])
            if len(rows) != COUNTS[name] or len({r['id'] for r in rows}) != len(rows):
                raise ValueError(f'Incomplete question inventory: {name}')
            validate_spec(spec, rows, context)
            if rank == 0:
                for path, digest in plan['identity']['sources'].items():
                    if sha256(path) != digest:
                        raise ValueError(f'Benchmark reference changed: {path}')
            identity = dict(common, rgb_plan_sha256=sha256(plan_path), benchmark=name,
                feature_index_sha256=sha256(cache/f'index/{context}/{name}.json'),
                manifest_sha256=spec['manifest_sha256'], max_sequence_length=context,
                visual_budget_fraction=.8, visual_token_budget=visual_budget(context), fps=2., max_frames=2048,
                max_pixels=640*640)
            target = directory/name
            target.mkdir(exist_ok=True)
            indices = assignments(spec, world)[rank]
            done = broadcast(cached_completion(target, identity) if rank == 0 else None, world)
            if not done:
                missing = [i for i in indices if read_video(video_path(target, spec['videos'][i]),
                                                           identity, spec['videos'][i]) is None]
                start = time.monotonic()
                if missing:
                    if model is None:
                        model = load_cached_model(checkpoint, device)
                        processor = load_processor(config['base'])
                        del checkpoint
                    run_config = dict(config, max_sequence_length=context, rgb_chunk_frames=32)
                    dataset = CachedEvaluationVideos(cache_specs[(context, name)], spec, missing)
                    loader = make_loader(dataset, single_video_collate, a.workers, batch_size=1)
                    def complete_video(video, payload):
                        save_json(dict(identity=identity, video_key=video['video_key'], payload=payload,
                                       payload_sha256=digest_json(payload)), video_path(target, video))
                    predict_videos(model, loader, spec, rows, run_config, device, processor,
                                   on_video=complete_video)
                    del loader, dataset
                torch.cuda.synchronize()
                seconds = torch.tensor(time.monotonic()-start, device=device)
                if world > 1:
                    dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
                barrier(world)
                if rank == 0:
                    merged, stats = [], []
                    for video in spec['videos']:
                        value = read_video(video_path(target, video), identity, video)
                        if value is None:
                            raise ValueError(f'Missing video output: {video["video_key"]}')
                        merged.extend(value['predictions'])
                        stats.append(value['cache'])
                    by_id = {r['id']: r for r in merged}
                    if len(merged) != len(rows) or set(by_id) != {r['id'] for r in rows}:
                        raise ValueError(f'Missing/duplicate benchmark predictions: {name}')
                    merged = [by_id[r['id']] for r in rows]
                    metrics = score_qa(name, merged, rows, spec, target)
                    metrics.update(identity, n=len(rows), videos=len(spec['videos']), world_size=world,
                        seconds_this_attempt=float(seconds), use_subtitles=False, use_audio=False,
                        prompt_protocol='independent questions; private KV/DeltaNet forks',
                        total_video_compute_seconds=sum(s['seconds'] for s in stats))
                    atomic_save(dict(scores=merged, checkpoint_id=checkpoint_id), target/'scores.pt')
                    save_json(stats, target/'video_cache.json')
                    save_json(metrics, target/'metrics.json')
                    files = [target/file for file in ('scores.pt', 'metrics.json', 'answers.json',
                                                      'predictions.jsonl', 'video_cache.json')]
                    if name == 'videomme':
                        files.append(target/'official_results.json')
                    mark_complete(target, identity, files, n=len(rows))
                    print(json.dumps(dict(event='benchmark_complete', context=context,
                                          benchmark=name, n=len(rows))), flush=True)
            barrier(world)
            if rank == 0:
                summary[name] = read(target/'metrics.json')
        if rank == 0:
            save_json(summary, directory/'summary.json')
            qa_results = summary
            mark_complete(directory, dict(common, context=context), [directory/'summary.json',
                *(directory/name/'completion.json' for name in COUNTS)], questions=sum(COUNTS.values()))
    if rank == 0:
        after = checkpoint_stat(checkpoint_path)
        if before != after:
            raise ValueError('Checkpoint changed during evaluation')
        save_json(qa_results, output/'summary.json')
        from timejev.qa_eval_settings import write_qa_results
        write_qa_results(qa_results, output)
        mark_complete(output, common, [output/'summary.json', output/'RESULTS.md', output/'qa_results.json',
            *(output/f'context_{c}/completion.json' for c in CONTEXTS)], contexts=list(CONTEXTS),
            benchmarks=list(COUNTS), questions_per_context=sum(COUNTS.values()))
    barrier(world)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
