"""One ZeRO-3 model, original grounding plus QA at 196608 and 262144 tokens."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.distributed as dist

from scripts.evaluate import COUNTS, QA_NAMES, cached_completion, mark_complete, read, score_qa
from scripts.train_qwen36 import setup
from timejev.benchmark import NAMES, select_decoder, export
from timejev.context_eval import CONTEXTS, validate_spec
from timejev.data import read_jsonl
from timejev.decision_inputs import CHOICE, FORMULATION, load_processor, task_of
from timejev.engine import atomic_save, make_loader, save_json, sha256
from timejev.multiturn_data import MultiturnCollator
from timejev.qwen36_checkpoint import load_exported_task_checkpoint, _complete_manifest
from timejev.qwen36_data import PROTOCOL, CAPACITY, evaluation_schedule, text_item
from timejev.qwen36_model import build_grounder, build_frozen_vision, encode_rgb
from timejev.qwen36_recipe import validate_recipe
from timejev.rgb_data import RGBEvaluationVideos, single_video_collate
from timejev.token_metadata import tokenizer_identity
from timejev.vision import VISUAL_PROTOCOL


def checkpoint_fingerprint(directory):
    report = _complete_manifest(directory)
    if not report or report.get('head_only'):
        raise ValueError('Evaluation requires a completed full task export')
    files = {n: sha256(directory/n) for n in sorted(report['files'])}
    files['complete_manifest.json'] = sha256(directory/'complete_manifest.json')
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


@torch.inference_mode()
def predict_packs(engine, visual, spec, rows, config, capacity, rank, world, device, processor, target):
    from timejev.multiturn_evaluation import query_chunks, PROTOCOL as EVAL_PROTOCOL
    from timejev.rgb_media import timeline
    indices, _ = evaluation_schedule(spec, capacity, rank, world)
    jobs = []
    for index in indices:
        video = spec['videos'][index]
        members = [rows[q['query_index']] for q in video['questions']]
        if [r['id'] for r in members] != [q['id'] for q in video['questions']]:
            raise ValueError('RGB evaluation identity changed')
        record = timeline(dict(video['record'], **video['geometry']))
        for chunk in query_chunks(processor, record, members, capacity):
            jobs.append((index, members[chunk['start']:chunk['end']], chunk))
    rounds = torch.tensor(len(jobs), dtype=torch.long, device=device)
    dist.all_reduce(rounds, op=dist.ReduceOp.MAX)
    rounds = int(rounds)
    loader = make_loader(RGBEvaluationVideos(config, spec, indices), single_video_collate,
                         config['workers'], batch_size=1)
    iterator = iter(loader)
    batchify = MultiturnCollator(processor.tokenizer, capacity)
    results, current_index, current_features, encodes = [], None, None, 0
    start = time.monotonic()
    keys = ('input_ids', 'mm_token_type_ids', 'video_grid_thw', 'pair_indices', 'query_indices',
            'score_mask', 'decision_tasks', 'episode_lengths', 'packed_sample_lengths', 'turn_episode_indices')
    for step in range(rounds):
        item = None
        if step < len(jobs):
            index, members, chunk = jobs[step]
            video = spec['videos'][index]
            if index != current_index:
                payload = next(iterator)
                if payload['index'] != index:
                    raise ValueError('RGB loader and collective schedule disagree')
                current_features = encode_rgb(visual, [payload['video_pixel_chunks']],
                    torch.tensor([video['geometry']['grid_thw']]), device)
                current_index = index
                encodes += 1
                del payload
            item = text_item(processor, members, video['record'],
                dict(id=members[0]['id'], geometry=video['geometry'], tokens=chunk['tokens']), capacity)
        batch = batchify([item] if item else [dict(empty_pack=True)])
        inputs = {k: batch[k].to(device) if torch.is_tensor(batch[k]) else batch[k] for k in keys if k in batch}
        inputs['rgb_visual_features'] = (current_features if item else
            torch.empty((0, visual.config.out_hidden_size), dtype=torch.bfloat16, device=device))
        output = engine(**inputs)
        logits, probabilities = output['logits'].float().cpu(), output['probabilities'].float().cpu()
        for i, row in enumerate(item['rows'] if item else []):
            n = int(batch['score_mask'][i].sum())
            if not torch.isfinite(logits[i, :n]).all() or not torch.isfinite(probabilities[i, :n]).all():
                raise ValueError(f'Nonfinite prediction: {row["id"]}')
            result = dict(id=row['id'], task=row.get('task', 'grounding'), formulation=FORMULATION,
                logits=logits[i, :n].tolist(), probabilities=probabilities[i, :n].tolist(),
                audit=dict(item['audit'], protocol=EVAL_PROTOCOL, original_video_turn_index=chunk['start']+i))
            if task_of(row) == CHOICE:
                answer = int(logits[i, :n].argmax())
                result.update(answer_index=answer, answer=row['options'][answer], options=row['options'])
            else:
                result.update(times=item['times'].tolist(), bounds=item['audit']['score_bounds'],
                              sampled_duration=row['sampled_duration'])
            results.append(result)
        if step % 5 == 0 or step+1 == rounds:
            save_json(dict(rank=rank, completed_queries=len(results), collective_forwards=step+1,
                total_collective_forwards=rounds, elapsed_seconds=time.monotonic()-start),
                target/f'progress_rank{rank}.json')
        del batch, inputs, output
    if encodes != len(indices):
        raise ValueError('Each assigned video must be encoded exactly once')
    return results, dict(encoded_videos=encodes, questions=len(results), real_packs=len(jobs),
        collective_forwards=rounds, empty_forwards=rounds-len(jobs), language_cache=False,
        evaluation_protocol=EVAL_PROTOCOL)


def main():
    import deepspeed
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--plans', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    checkpoint, plans, out = Path(a.checkpoint).resolve(), Path(a.plans).resolve(), Path(a.out).resolve()
    metadata = read(checkpoint/'task_model_config.json')
    config, checkpoint_id = metadata['recipe'], metadata['checkpoint_id']
    validate_recipe(config)
    if config['training_stage'] != 'joint' or metadata['step'] < 1:
        raise ValueError('Expected the completed mixed joint checkpoint')
    rank, world, device = setup(config['seed'])
    if world != 64:
        raise ValueError('This evaluator requires 8 nodes x 8 GPUs')
    out.mkdir(parents=True, exist_ok=True)
    envelope = [checkpoint_fingerprint(checkpoint) if rank == 0 else None]
    dist.broadcast_object_list(envelope)
    checkpoint_stats = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in checkpoint.iterdir() if p.is_file()}
    root = Path(__file__).resolve().parents[1]
    code_hash = hashlib.sha256(json.dumps({str(p.relative_to(root)): sha256(p)
        for folder in ('timejev', 'scripts') for p in sorted((root/folder).glob('*.py'))}, sort_keys=True).encode()).hexdigest()
    processor = load_processor(config['base'])
    registry = read(config['qa_evaluation_config'])
    engine, visual, summary, completions = None, None, {}, []
    for capacity in (CAPACITY, *CONTEXTS):
        plan_path = plans/f'evaluation_{capacity}.json'
        plan = read(plan_path)
        names = NAMES if capacity == CAPACITY else QA_NAMES
        if (plan['protocol'] != PROTOCOL or plan['base'] != config['base'] or plan['context'] != capacity
                or set(plan['benchmarks']) != set(names)
                or plan['identity']['tokenizer'] != tokenizer_identity(processor, prompt_version=PROTOCOL)):
            raise ValueError('Evaluation tokenizer/protocol/context differs from checkpoint')
        group = out/('grounding' if capacity == CAPACITY else f'qa_{capacity}')
        group.mkdir(exist_ok=True)
        all_rows, all_scores, timing = {}, {}, {}
        for name in names:
            qa, spec = name in QA_NAMES, plan['benchmarks'][name]
            rows = read_jsonl(spec['manifest'])
            if (len(rows) != COUNTS[name] or len({r['id'] for r in rows}) != len(rows)
                    or sha256(spec['manifest']) != spec['manifest_sha256']):
                raise ValueError(f'Incomplete/changed benchmark: {name}')
            if qa:
                validate_spec(spec, rows, capacity)
            reference = registry[name]['2'] if qa else None
            identity = dict(evaluation_protocol='greedy_video_query_chunks_single_forward_v1', checkpoint_id=checkpoint_id, checkpoint_sha256=envelope[0], code_sha256=code_hash,
                manifest_sha256=spec['manifest_sha256'], rgb_plan_sha256=sha256(plan_path),
                max_sequence_length=capacity, fps=2. if qa else 4., benchmark=name, input_mode=PROTOCOL)
            if qa:
                identity['official_reference_sha256'] = sha256(reference.get('official_questions', reference.get('lvbench_metadata')))
            target = group/name
            target.mkdir(exist_ok=True)
            done = [cached_completion(target, identity) if rank == 0 else None]
            dist.broadcast_object_list(done)
            if not done[0]:
                if engine is None:
                    ds_config = read(config['deepspeed'])
                    model = build_grounder(config, device, ds_config=ds_config, load_pretrained=False)
                    engine, _, _, _ = deepspeed.initialize(model=model, config=ds_config, dist_init_required=False)
                    # The head is created outside zero.Init; engine initialization
                    # partitions it before the strict all-parameter ZeRO loader.
                    load_exported_task_checkpoint(model, checkpoint, zero3=True)
                    engine.eval()
                    visual = build_frozen_vision(config, device)
                start = time.monotonic()
                predictions, stats = predict_packs(engine, visual, spec, rows, config, capacity,
                    rank, world, device, processor, target)
                torch.cuda.synchronize()
                seconds = torch.tensor(time.monotonic()-start, device=device)
                atomic_save(predictions, target/f'rank{rank}.pt')
                save_json(stats, target/f'cache_rank{rank}.json')
                dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
                dist.barrier()
                if rank == 0:
                    merged = [r for worker in range(world) for r in torch.load(target/f'rank{worker}.pt', weights_only=True)]
                    by_id = {r['id']: r for r in merged}
                    if len(merged) != len(rows) or set(by_id) != {r['id'] for r in rows}:
                        raise ValueError(f'Missing or duplicate predictions: {name}')
                    merged = [by_id[r['id']] for r in rows]
                    stats = dict(n=len(rows), videos=len(spec['videos']), seconds=float(seconds), world_size=world,
                        sample_fps=2. if qa else 4., max_frames=2048, max_sequence_length=capacity,
                        visual_budget_fraction=.8, checkpoint_id=checkpoint_id, input_mode=PROTOCOL,
                        queries_per_second=len(rows)/max(float(seconds), 1e-9),
                        shards=[read(target/f'cache_rank{worker}.json') for worker in range(world)])
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
                    print(json.dumps(dict(event='benchmark_complete', benchmark=name, context=capacity,
                                          questions=len(rows), seconds=float(seconds))), flush=True)
            dist.barrier()
            if rank == 0:
                completions.append(target/'completion.json')
                if qa:
                    summary[f'{name}_{capacity}'] = read(target/'metrics.json')
                else:
                    blob = torch.load(target/'scores.pt', weights_only=True)
                    all_rows[name], all_scores[name], timing[name] = rows, blob['scores'], blob['timing']
        if rank == 0 and capacity == CAPACITY:
            decoder = select_decoder(all_rows, all_scores, config['decoder_thresholds'], config['decoder_smoothing'],
                postprocess=config['interval_postprocess'], selection_postprocess='none')
            decoder.update(checkpoint_id=checkpoint_id, visual_protocol=VISUAL_PROTOCOL, sample_fps=4.)
            summary.update(export(all_rows, all_scores, decoder, group/'vtg', timing))
        if rank == 0:
            save_json(summary, out/'summary.json')
        dist.barrier()
    if rank == 0:
        if len(summary) != 11:
            raise ValueError('Expected three grounding and eight QA results')
        after = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in checkpoint.iterdir() if p.is_file()}
        if checkpoint_stats != after:
            raise ValueError('Checkpoint changed during evaluation')
        from timejev.qa_eval_settings import write_context_results
        comparison = {str(c): {n: summary[f'{n}_{c}'] for n in QA_NAMES} for c in (196608, 262144)}
        selected_files = write_context_results(comparison, out, folder_pattern='qa_{context}')
        files = [out/'summary.json', *completions, *selected_files, *((out/'grounding/vtg').glob('*.json')),
                 *((out/'grounding/vtg').glob('*.jsonl'))]
        mark_complete(out, dict(checkpoint_id=checkpoint_id, checkpoint_sha256=envelope[0], code_sha256=code_hash),
                      files, benchmarks=list(summary))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
