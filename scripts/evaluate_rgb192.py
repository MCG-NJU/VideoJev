"""Seven benchmarks from RGB with 192K contexts and causal multi-query groups."""
import argparse
from timejev.hf_checkpoint import load_training_checkpoint, checkpoint_fingerprint, checkpoint_stat
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist

from scripts.evaluate import (COUNTS, QA_NAMES, barrier, cached_completion, mark_complete, read, score_qa)
from timejev.benchmark import NAMES, select_decoder, export
from timejev.data import read_jsonl, to_device
from timejev.decision_inputs import CHOICE, FORMULATION, load_processor, task_of
from timejev.decision_model import from_config
from timejev.engine import atomic_save, make_loader, save_json, setup, sha256
from timejev.multiturn_data import MultiturnCollator
from timejev.rgb_data import (PROTOCOL, RGBEvaluationVideos, single_video_collate, text_item)
from timejev.vision import VISUAL_PROTOCOL


@torch.inference_mode()
def predict_videos(model, loader, spec, rows, config, device, processor):
    from timejev.multiturn_evaluation import predict_video
    from timejev.rgb_media import timeline
    results = []
    stats = dict(encoded_videos=0, questions=0, suffix_forwards=0, hits=0, misses=0)
    model.eval()
    for payload in loader:
        video = spec['videos'][payload['index']]
        grid = torch.tensor([video['geometry']['grid_thw']], device=device, dtype=torch.long)
        features = model.backbone.visual([payload['video_pixel_chunks']], grid_thw=grid)
        members = [rows[q['query_index']] for q in video['questions']]
        if [r['id'] for r in members] != [q['id'] for q in video['questions']]:
            raise ValueError('Evaluation question identity differs from the RGB plan')
        record = timeline(dict(video['record'], **video['geometry'], cache_id=PROTOCOL))
        output = predict_video(model, processor, features, record, members,
                               config['max_sequence_length'], device)
        results.extend(output['predictions'])
        stats['encoded_videos'] += 1
        stats['questions'] += len(members)
        stats['suffix_forwards'] += output['forward_calls']
        stats['misses'] += 1
        del features, payload
    return results, stats


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--evaluation-plan', help='Audited replacement RGB plan; leaves the trained checkpoint unchanged')
    parser.add_argument('--benchmark', choices=('all', 'vtg', *QA_NAMES), default='all')
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint).resolve(strict=True)
    before = checkpoint_stat(checkpoint_path)
    checkpoint = load_training_checkpoint(checkpoint_path)
    config, checkpoint_id = dict(checkpoint['config']), checkpoint['checkpoint_id']
    if args.evaluation_plan:
        config['rgb_evaluation'] = str(Path(args.evaluation_plan).resolve(strict=True))
    if config.get('input_mode') != 'rgb_dynamic' or config['max_sequence_length'] != 196608:
        raise ValueError('Expected the RGB 192K checkpoint')
    if config.get('training_stage') != 'joint' or checkpoint['step'] <= 0:
        raise ValueError('Expected a completed joint-stage checkpoint')
    rank, world, device = setup(config['seed'])
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    envelope = [checkpoint_fingerprint(checkpoint_path) if rank == 0 else None]
    if world > 1:
        dist.broadcast_object_list(envelope)
    plan = read(config['rgb_evaluation'])
    if args.evaluation_plan and (plan.get('protocol') != PROTOCOL or
                                 plan.get('max_sequence_length') != 196608):
        raise ValueError('Replacement evaluation plan must use the RGB 192K protocol')
    registry = read(config['qa_evaluation_config'])
    names = (*NAMES, *QA_NAMES) if args.benchmark == 'all' else NAMES if args.benchmark == 'vtg' else (args.benchmark,)
    code_root = Path(__file__).resolve().parents[1]
    code_hash = hashlib.sha256(json.dumps({str(p.relative_to(code_root)): sha256(p)
        for folder in ('timejev', 'scripts') for p in sorted((code_root/folder).glob('*.py'))}, sort_keys=True).encode()).hexdigest()
    model, processor, all_rows, all_scores, timing = None, None, {}, {}, {}
    for name in names:
        qa, spec = name in QA_NAMES, plan['benchmarks'][name]
        rows = read_jsonl(spec['manifest'])
        if (len(rows) != COUNTS[name] or len({r['id'] for r in rows}) != COUNTS[name]
                or sha256(spec['manifest']) != spec['manifest_sha256']):
            raise ValueError(f'Incomplete/changed reference manifest: {name}')
        reference = registry[name]['2'] if qa else None
        identity = dict(evaluation_protocol='greedy_video_query_chunks_single_forward_v1', checkpoint_id=checkpoint_id, checkpoint_sha256=envelope[0], code_sha256=code_hash,
            manifest_sha256=spec['manifest_sha256'], rgb_plan_sha256=sha256(config['rgb_evaluation']),
            max_sequence_length=196608, fps=2. if qa else 4., benchmark=name, input_mode=PROTOCOL)
        if qa:
            identity['official_reference_sha256'] = sha256(reference.get('official_questions', reference.get('lvbench_metadata')))
        target = output/name
        target.mkdir(exist_ok=True)
        done = [cached_completion(target, identity) if rank == 0 else None]
        if world > 1:
            dist.broadcast_object_list(done)
        if not done[0]:
            if model is None:
                model = from_config(config, checkpointing=False)
                model.load_checkpoint(checkpoint)
                model.to(device).eval()
                processor = load_processor(config['base'])
                del checkpoint
            indices = sorted(range(len(spec['videos'])), key=lambda i: spec['videos'][i]['video_key'])[rank::world]
            dataset = RGBEvaluationVideos(config, spec, indices)
            loader = make_loader(dataset, single_video_collate, config['workers'], batch_size=1)
            torch.cuda.synchronize()
            start = time.monotonic()
            predictions, cache = predict_videos(model, loader, spec, rows, config, device, processor)
            torch.cuda.synchronize()
            seconds = torch.tensor(time.monotonic()-start, device=device)
            atomic_save(predictions, target/f'rank{rank}.pt')
            save_json(cache, target/f'cache_rank{rank}.json')
            if world > 1:
                dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
            barrier(world)
            if rank == 0:
                merged = [p for worker in range(world) for p in torch.load(target/f'rank{worker}.pt', weights_only=True)]
                by_id = {p['id']: p for p in merged}
                if len(merged) != len(rows) or set(by_id) != {r['id'] for r in rows}:
                    raise ValueError(f'Missing/duplicate benchmark predictions: {name}')
                merged = [by_id[r['id']] for r in rows]
                stats = dict(n=len(rows), videos=len(spec['videos']), seconds=float(seconds), world_size=world,
                    sample_fps=2. if qa else 4., max_frames=2048, max_sequence_length=196608,
                    checkpoint_id=checkpoint_id, queries_per_second=len(rows)/max(float(seconds), 1e-9),
                    input_mode=PROTOCOL, prompt_protocol='independent questions; private KV/DeltaNet forks',
                    video_cache=[read(target/f'cache_rank{worker}.json') for worker in range(world)])
                atomic_save(dict(scores=merged, timing=stats, checkpoint_id=checkpoint_id,
                    visual_protocol=VISUAL_PROTOCOL, manifest_sha256=spec['manifest_sha256']), target/'scores.pt')
                files = [target/'scores.pt']
                if qa:
                    metrics = score_qa(name, merged, rows, reference, target)
                    metrics.update(stats, use_subtitles=False, use_audio=False)
                    save_json(metrics, target/'metrics.json')
                    files += [target/'metrics.json', target/'answers.json', target/'predictions.jsonl']
                    if name == 'videomme':
                        files.append(target/'official_results.json')
                mark_complete(target, identity, files, n=len(rows))
                print(json.dumps(dict(event='benchmark_complete', benchmark=name, n=len(rows), seconds=float(seconds))), flush=True)
            del loader, dataset
        barrier(world)
        if rank == 0 and not qa:
            blob = torch.load(target/'scores.pt', weights_only=True)
            all_rows[name], all_scores[name], timing[name] = rows, blob['scores'], blob['timing']
    if rank == 0:
        summary = {}
        if all(name in all_scores for name in NAMES):
            decoder = select_decoder(all_rows, all_scores, config['decoder_thresholds'], config['decoder_smoothing'])
            decoder.update(checkpoint_id=checkpoint_id, visual_protocol=VISUAL_PROTOCOL, sample_fps=4.)
            summary.update(export(all_rows, all_scores, decoder, output/'vtg', timing))
        for name in QA_NAMES:
            if name in names:
                summary[name] = read(output/name/'metrics.json')
        save_json(summary, output/('summary.json' if args.benchmark == 'all' else f'{args.benchmark}_summary.json'))
        after = checkpoint_stat(checkpoint_path)
        if before != after:
            raise ValueError('Checkpoint changed during RGB evaluation')
        if args.benchmark == 'all':
            files = [output/'summary.json', *(output/name/'completion.json' for name in names),
                     *((output/'vtg').glob('*.json')), *((output/'vtg').glob('*.jsonl'))]
            mark_complete(output, dict(checkpoint_id=checkpoint_id, checkpoint_sha256=envelope[0],
                code_sha256=code_hash), files, benchmarks=list(names), questions=sum(COUNTS.values()))
    barrier(world)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
