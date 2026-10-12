"""Evaluate a single joint checkpoint on VTG (4fps) and video MCQ (2fps).

Completed benchmark outputs are reused only after verifying their input identity
and output hashes. Each GPU owns complete videos and evaluates their questions in causal groups.
"""
from timejev.config import read_config
import argparse
from timejev.hf_checkpoint import load_training_checkpoint, checkpoint_fingerprint, checkpoint_stat
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist

from timejev.benchmark import NAMES, select_decoder, export
from timejev.data import read_jsonl
from timejev.decision_benchmark import predict_decisions
from timejev.decision_io import data_module
from timejev.decision_model import from_config
from timejev.engine import atomic_save, make_loader, predict, save_json, setup, sha256
from timejev.feature_sources import register_training_caches
from timejev.features import validate_cache
from timejev.qa_eval_settings import resolve_evaluation, apply_evaluation, validate_evaluation
from timejev.timestamp_option_inputs import load_processor
from timejev.vision import VISUAL_PROTOCOL

QA_NAMES = ('lvbench', 'videomme', 'longvideobench', 'mlvu')
COUNTS = dict(charades=3363, activitynet=4500, qvhighlights=1541,
              lvbench=1549, videomme=2700, longvideobench=1337, mlvu=2174)


def read(path):
    return read_config(path)


def barrier(world):
    if world > 1:
        dist.barrier()


def cached_completion(output, identity):
    path = output / 'completion.json'
    if not path.exists():
        return False
    state = read(path)
    if state.get('identity') != identity:
        raise ValueError(f'Evaluation identity changed; choose a new output directory: {output}')
    if not state.get('complete') or not state.get('files'):
        return False
    for name, digest in state['files'].items():
        if not (output / name).is_file() or sha256(output / name) != digest:
            raise ValueError(f'Completed evaluation artifact changed: {output/name}')
    return True


def mark_complete(output, identity, files, **metadata):
    save_json(dict(complete=True, identity=identity, **metadata,
                   files={str(p.relative_to(output)): sha256(p) for p in files}),
              output / 'completion.json')


def score_qa(name, predictions, rows, evaluation, output):
    if name == 'lvbench':
        from timejev.qa_benchmark import score_lvbench
        metrics, answers, scored = score_lvbench(predictions, rows, evaluation['lvbench_metadata'])
    else:
        reference = read(evaluation['official_questions'])
        if name == 'videomme':
            from timejev.videomme_benchmark import score_videomme
            metrics, answers, scored, official = score_videomme(predictions, rows, reference)
            save_json(official, output/'official_results.json')
        elif name == 'longvideobench':
            from timejev.longvideobench_benchmark import score_longvideobench
            metrics, answers, scored = score_longvideobench(predictions, rows, reference)
        else:
            from timejev.mlvu_benchmark import score_mlvu
            metrics, answers, scored = score_mlvu(predictions, rows, reference)
    save_json(answers, output/'answers.json')
    (output/'predictions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in scored))
    return metrics


@torch.inference_mode()
def predict_dataset(model, processor, dataset, indices, config, device):
    from timejev.features import read_features
    from timejev.adaptive_features import read_adaptive_features
    from timejev.multiturn_evaluation import predict_video
    grouped = {}
    for index in indices:
        row = dataset.rows[index]
        grouped.setdefault(row['video_key'], []).append(row)
    results, forwards = [], 0
    for members in grouped.values():
        if dataset.feature_variant_root:
            features, record = read_adaptive_features(dataset.feature_root, dataset.feature_variant_root,
                members[0], dataset.fps, dataset.feature_spec['cache_id'], dataset.variant_spec, dataset.variant_plan)
        else:
            features, record = read_features(dataset.feature_root, members[0], dataset.fps,
                                             dataset.feature_spec['cache_id'])
        output = predict_video(model, processor, features, record, members,
                               config['max_sequence_length'], device)
        results.extend(output['predictions'])
        forwards += output['forward_calls']
    model.last_video_cache_stats = dict(hits=0, misses=len(grouped), queries=forwards,
                                       forward_calls=forwards, videos=len(grouped))
    return results


@torch.inference_mode()
def main():
    from timejev.hf_evaluation import released_checkpoint_cli, main as evaluate_hf
    if released_checkpoint_cli():
        return evaluate_hf()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--benchmark', choices=('all', 'vtg', *QA_NAMES), default='all')
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint).resolve(strict=True)
    before = checkpoint_stat(checkpoint_path)
    checkpoint = load_training_checkpoint(checkpoint_path)
    c, checkpoint_id = dict(checkpoint['config']), checkpoint['checkpoint_id']
    if c.get('training_stage') != 'joint' or checkpoint['step'] <= 0:
        raise ValueError('Expected a completed joint-stage checkpoint')
    rank, world, device = setup(c['seed'])
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    # Hash only once across all workers. Shared storage need not stream 64 copies.
    envelope = [checkpoint_fingerprint(checkpoint_path) if rank == 0 else None]
    if world > 1:
        dist.broadcast_object_list(envelope)
    model = None
    all_rows, all_scores, timing = {}, {}, {}
    names = (*NAMES, *QA_NAMES) if args.benchmark == 'all' else NAMES if args.benchmark == 'vtg' else (args.benchmark,)
    code_root = Path(__file__).resolve().parents[1]
    code_hash = hashlib.sha256(json.dumps({str(p.relative_to(code_root)): sha256(p)
        for folder in ('timejev', 'scripts') for p in sorted((code_root/folder).glob('*.py'))}, sort_keys=True).encode()).hexdigest()
    for name in names:
        qa = name in QA_NAMES
        evaluation = resolve_evaluation(name, 2.) if qa else None
        config = apply_evaluation(c, evaluation) if qa else c
        manifest = config['benchmarks'][name]
        rows = read_jsonl(manifest)
        if len(rows) != COUNTS[name] or len({r['id'] for r in rows}) != COUNTS[name]:
            raise ValueError(f'Incomplete reference manifest: {name}')
        identity = dict(evaluation_protocol='greedy_video_query_chunks_single_forward_v1', checkpoint_id=checkpoint_id, checkpoint_sha256=envelope[0], code_sha256=code_hash,
            manifest_sha256=sha256(manifest), lengths_sha256=sha256(config['packed_eval_lengths'][name]),
            fps=config['eval_fps'], benchmark=name,
            cache_provenance_sha256=sha256(Path(config['feature_root'])/'provenance.json'))
        if qa:
            identity['official_reference_sha256'] = sha256(evaluation.get('official_questions', evaluation.get('lvbench_metadata')))
        else:
            identity['variant_provenance_sha256'] = sha256(Path(config['feature_variant_root'])/'provenance.json')
        target = output/name
        target.mkdir(exist_ok=True)
        done = [cached_completion(target, identity) if rank == 0 else None]
        if world > 1:
            dist.broadcast_object_list(done)
        if not done[0]:
            spec = validate_evaluation(config, evaluation) if qa else validate_cache(
                config['feature_root'], [manifest], config['eval_fps'], replacement_root=config.get('feature_variant_root'))
            if model is None:
                model = from_config(c, checkpointing=False)
                model.load_checkpoint(checkpoint)
                model.to(device).eval()
                del checkpoint
            register_training_caches(model, spec)
            if config.get('feature_variant_root'):
                register_training_caches(model, validate_cache(config['feature_variant_root'], [],
                                         config['eval_fps'], require_complete=False))
            inputs = data_module(config)
            dataset = inputs.DecisionDataset(manifest, config['base'], fps=config['eval_fps'],
                query_placement='last', feature_root=config['feature_root'],
                feature_variant_root=config.get('feature_variant_root'), fast_feature_inputs=True)
            dataset.lengths = inputs.load_lengths(config['packed_eval_lengths'][name], manifest, config, dataset.rows)
            videos = sorted({r['video_key'] for r in rows})
            owners = {video: i % world for i, video in enumerate(videos)}
            indices = sorted((i for i, r in enumerate(rows) if owners[r['video_key']] == rank),
                             key=lambda i: (rows[i]['video_key'], i))
            processor = load_processor(c['base'])
            torch.cuda.synchronize()
            started = time.monotonic()
            predictions = predict_dataset(model, processor, dataset, indices, config, device)
            torch.cuda.synchronize()
            seconds = torch.tensor(time.monotonic()-started, device=device)
            atomic_save(predictions, target/f'rank{rank}.pt')
            save_json(model.last_video_cache_stats, target/f'cache_rank{rank}.json')
            if world > 1:
                dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
            barrier(world)
            if rank == 0:
                merged = [p for worker in range(world) for p in torch.load(target/f'rank{worker}.pt', weights_only=True)]
                by_id = {p['id']: p for p in merged}
                if len(merged) != len(rows) or len(by_id) != len(rows) or set(by_id) != {r['id'] for r in rows}:
                    raise ValueError(f'Missing/duplicate predictions: {name}')
                merged = [by_id[r['id']] for r in rows]
                stats = dict(n=len(rows), videos=len(videos), seconds=float(seconds), world_size=world,
                    sample_fps=config['eval_fps'], max_sequence_length=c['max_sequence_length'],
                    checkpoint_id=checkpoint_id, queries_per_second=len(rows)/max(float(seconds), 1e-9),
                    prompt_protocol='greedy_video_query_chunks_single_forward_v1',
                    video_cache=[read(target/f'cache_rank{worker}.json') for worker in range(world)])
                atomic_save(dict(scores=merged, timing=stats, checkpoint_id=checkpoint_id,
                    visual_protocol=VISUAL_PROTOCOL, manifest_sha256=identity['manifest_sha256']), target/'scores.pt')
                files = [target/'scores.pt']
                if qa:
                    metrics = score_qa(name, merged, rows, evaluation, target)
                    metrics.update(stats, max_frames=2048, use_subtitles=False, use_audio=False)
                    save_json(metrics, target/'metrics.json')
                    files += [target/'metrics.json', target/'answers.json', target/'predictions.jsonl']
                    if name == 'videomme':
                        files.append(target/'official_results.json')
                mark_complete(target, identity, files, n=len(rows))
                print(json.dumps(dict(event='benchmark_complete', benchmark=name, n=len(rows), seconds=float(seconds))), flush=True)
            del dataset
        barrier(world)
        if rank == 0 and not qa:
            blob = torch.load(target/'scores.pt', weights_only=True)
            all_rows[name], all_scores[name], timing[name] = rows, blob['scores'], blob['timing']
    if rank == 0:
        summary = {}
        if all(n in all_scores for n in NAMES):
            decoder = select_decoder(all_rows, all_scores, c['decoder_thresholds'], c['decoder_smoothing'])
            decoder.update(checkpoint_id=checkpoint_id, visual_protocol=VISUAL_PROTOCOL, sample_fps=4.)
            vtg = export(all_rows, all_scores, decoder, output/'vtg', timing)
            summary.update(vtg)
        for name in QA_NAMES:
            if name in names:
                summary[name] = read(output/name/'metrics.json')
        summary_name = 'summary.json' if args.benchmark == 'all' else f'{args.benchmark}_summary.json'
        save_json(summary, output/summary_name)
        after = checkpoint_stat(checkpoint_path)
        if before != after:
            raise ValueError('Checkpoint changed during evaluation')
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
