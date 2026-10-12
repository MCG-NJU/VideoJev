"""Evaluate released HF checkpoints with labeled JSONL manifests."""
import argparse
import json
from pathlib import Path

from .qa_eval_settings import QA_CONTEXT

VTG = ('charades', 'activitynet', 'qvhighlights')
QA = ('lvbench', 'videomme', 'longvideobench', 'mlvu')
COUNTS = dict(charades=3363, activitynet=4500, qvhighlights=1541,
              lvbench=1549, videomme=2700, longvideobench=1337, mlvu=2174)


def released_checkpoint_cli(argv=None):
    """Dispatch public HF packages without changing training-checkpoint recovery."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--checkpoint')
    args, _ = parser.parse_known_args(argv)
    if not args.checkpoint:
        return False
    path = Path(args.checkpoint)
    if not path.exists():
        return not path.is_absolute() and len(args.checkpoint.split('/')) == 2
    if not path.is_dir() or not (path/'config.json').is_file():
        return False
    return (json.loads((path/'config.json').read_text()).get('model_type') == 'videojev'
            and not (path/'training_state.json').exists())


def checkpoint_directory(value, revision=None):
    path = Path(value)
    if path.is_dir():
        return path.resolve()
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(value, revision=revision))


def evaluation_inputs(args):
    from .user_data import load_manifest
    from .engine import sha256
    if args.manifest:
        if args.benchmark in ('all', 'vtg'):
            names = ('custom',)
        else:
            names = (args.benchmark,)
    else:
        if args.benchmark == 'custom':
            raise ValueError('--benchmark custom requires --manifest')
        names = (*VTG, *QA) if args.benchmark == 'all' else VTG if args.benchmark == 'vtg' else (args.benchmark,)
    result = {}
    for name in names:
        spec = None
        if args.plans:
            context = QA_CONTEXT if name in QA else args.grounding_context
            plan_path = Path(args.plans)/f'evaluation_{context}.json'
            plan = json.loads(plan_path.read_text())
            if plan['context'] != context:
                raise ValueError('Plan context differs from evaluation settings')
            spec = plan['benchmarks'][name]
            manifest = Path(spec['manifest'])
            if not manifest.is_absolute():
                manifest = plan_path.parent/manifest
            if sha256(manifest) != spec['manifest_sha256']:
                raise ValueError('Evaluation manifest changed since planning')
        else:
            manifest = Path(args.manifest) if args.manifest else Path(args.data_root)/name/'questions.jsonl'
        rows = load_manifest(manifest)
        reference = None
        if name != 'custom':
            expected_task = 'choice' if name in QA else 'grounding'
            if len(rows) != COUNTS[name] or any(r['task'] != expected_task for r in rows):
                raise ValueError(f'{name} requires all {COUNTS[name]} {expected_task} questions')
        if name in QA:
            reference = Path(args.reference) if args.reference else manifest.parent/('official.jsonl' if name == 'lvbench' else 'official.json')
            if not reference.is_file():
                raise FileNotFoundError(reference)
        if spec:
            indices = [q['query_index'] for v in spec['videos'] for q in v['questions']]
            if sorted(indices) != list(range(len(rows))):
                raise ValueError('Plan must cover every question exactly once')
            for video in spec['videos']:
                if any(rows[q['query_index']]['id'] != q['id'] or
                       rows[q['query_index']]['video_key'] != video['video_key'] for q in video['questions']):
                    raise ValueError('Question identity differs from evaluation plan')
        result[name] = dict(manifest=manifest.resolve(), rows=rows, reference=reference, plan=spec)
    return result


def score_predictions(name, predictions, rows, reference, out):
    from .engine import save_json
    from .temporal import metrics as grounding_metrics
    from .user_data import write_jsonl
    by_id = {p['id']: p for p in predictions}
    if len(by_id) != len(predictions) or set(by_id) != {r['id'] for r in rows}:
        raise ValueError('Missing, duplicate or unknown prediction IDs')
    ordered = [by_id[r['id']] for r in rows]
    if name in QA:
        # Official scorers check annotation identities and option ordering.
        from scripts.evaluate import score_qa
        annotated = [dict(p, options=r['options']) for p, r in zip(ordered, rows)]
        key = 'lvbench_metadata' if name == 'lvbench' else 'official_questions'
        report = score_qa(name, annotated, rows, {key: str(reference)}, out)
    else:
        report = {}
        grounding = [(r, p) for r, p in zip(rows, ordered) if r['task'] == 'grounding']
        choice = [(r, p) for r, p in zip(rows, ordered) if r['task'] == 'choice']
        if grounding:
            values = grounding_metrics([p['interval'] for _, p in grounding], [r for r, _ in grounding])
            if name in VTG:
                report.update(values)
            else:
                report['grounding'] = values
        if choice:
            correct = sum(r['answer_index'] == p['answer_index'] for r, p in choice)
            report['choice'] = dict(n=len(choice), correct=correct, accuracy=correct/len(choice), context=QA_CONTEXT)
        write_jsonl(ordered, out/'predictions.jsonl')
    report['n'] = len(rows)
    if name in QA:
        report['context'] = QA_CONTEXT
    save_json(report, out/'metrics.json')
    return report


def predict_rows(model, processor, rows, grounding_context, threshold, plan=None):
    """Bounded RGB preprocessing; labels never enter the model input."""
    import torch
    from .context_eval import choose_geometry
    from .multiturn_inputs import group_memberships
    from .rgb_data import geometry, text_lengths
    from .rgb_media import metadata, preprocess, timeline
    results = []
    # Sampling/context differ between tasks, even for a mixed custom manifest.
    for task in ('grounding', 'choice'):
        members = [r for r in rows if r['task'] == task]
        if not members:
            continue
        context = QA_CONTEXT if task == 'choice' else grounding_context
        for group in group_memberships(members):
            questions = [members[i] for i in group['query_indices']]
            if plan is not None:
                video = next(v for v in plan['videos'] if v['video_key'] == group['video_key'])
                record, shape = video['record'], video['geometry']
                if record['fps'] != (2. if task == 'choice' else 4.) or not 0 < record['frames'] <= 2048:
                    raise ValueError('Plan must use 2-FPS QA / 4-FPS grounding and at most 2048 frames')
            else:
                record = metadata(questions[0], 'lvbench' if task == 'choice' else 'charades')
                first, _ = text_lengths(processor, record, questions)
                if task == 'choice':
                    shape = choose_geometry(record, max(first)+63, context)
                else:
                    shape = geometry(record, max(max(first)+63, context-context*4//5), minimum=1024, capacity=context)
                    if shape is None:
                        raise ValueError(f'A complete question exceeds context: {group["video_key"]}')
            record = timeline(dict(record, **shape))
            chunks = preprocess(processor, record)
            features = []
            for chunk in chunks:
                h, w = record['grid_thw'][1:]
                grid = torch.tensor([[len(chunk)//(h*w), h, w]])
                features.append(model.encode_video(chunk, grid))
            features = torch.cat(features)
            inputs = [{k: q[k] for k in ('id', 'task', 'query', 'options') if k in q} for q in questions]
            results.extend(model.predict_features(processor, features, record, inputs,
                context=context, threshold=threshold))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, help='HF repository ID or local HF directory')
    parser.add_argument('--revision')
    parser.add_argument('--out', required=True)
    parser.add_argument('--benchmark', choices=('all', 'vtg', 'custom', *VTG, *QA), default='all')
    parser.add_argument('--data-root', default='data/benchmarks')
    parser.add_argument('--plans', help='Directory with evaluation_<context>.json RGB plans')
    parser.add_argument('--manifest', help='Labeled JSONL; omit --benchmark for custom data')
    parser.add_argument('--reference', help='Official annotation file for one QA benchmark')
    parser.add_argument('--grounding-context', type=int, default=147456)
    parser.add_argument('--threshold', type=float, default=.25)
    parser.add_argument('--dry-run', action='store_true', help='Validate manifests without loading weights')
    args = parser.parse_args(argv)
    if args.plans and args.manifest:
        parser.error('Use either --plans or --manifest')
    if not 0 < args.threshold < 1 or args.grounding_context < 64:
        parser.error('threshold must be in (0,1); grounding context must be at least 64')
    datasets = evaluation_inputs(args)
    if args.dry_run:
        print(json.dumps({n: dict(questions=len(v['rows']), qa_context=QA_CONTEXT) for n, v in datasets.items()}))
        return
    import torch
    import torch.distributed as dist
    from transformers import AutoModel, AutoProcessor
    from .engine import setup, save_json, sha256
    from .user_data import write_jsonl
    rank, world, device = setup(20260924)
    out = Path(args.out)
    if (out/'summary.json').exists():
        raise FileExistsError(f'Choose a new output directory: {out}')
    directory = checkpoint_directory(args.checkpoint, args.revision)
    native = json.loads((directory/'config.json').read_text())
    if native.get('model_type') != 'videojev':
        raise ValueError('Expected a VideoJev Hugging Face checkpoint')
    processor = AutoProcessor.from_pretrained(directory, trust_remote_code=True)
    model, loading = AutoModel.from_pretrained(directory, trust_remote_code=True, dtype='auto',
        device_map={'': str(device)}, attn_implementation='flash_attention_2', output_loading_info=True)
    if any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError(f'Incomplete checkpoint: {loading}')
    model.prepare_for_inference()
    summary = {}
    with torch.inference_mode():
        for name, dataset in datasets.items():
            target = out/name
            target.mkdir(parents=True, exist_ok=True)
            rows = dataset['rows']
            videos = sorted({r['video_key'] for r in rows})
            owned = set(videos[rank::world])
            local = [r for r in rows if r['video_key'] in owned]
            predictions = predict_rows(model, processor, local, args.grounding_context, args.threshold, dataset['plan'])
            write_jsonl(predictions, target/f'rank{rank}.jsonl')
            if world > 1:
                dist.barrier()
            if rank == 0:
                merged = [json.loads(line) for worker in range(world)
                          for line in (target/f'rank{worker}.jsonl').read_text().splitlines()]
                report = score_predictions(name, merged, rows, dataset['reference'], target)
                report.update(checkpoint_id=native.get('checkpoint_id'), manifest_sha256=sha256(dataset['manifest']),
                    grounding_context=args.grounding_context, qa_context=QA_CONTEXT,
                    threshold=args.threshold, max_frames=2048, grounding_fps=4., qa_fps=2.)
                save_json(report, target/'metrics.json')
                summary[name] = report
    if rank == 0:
        save_json(summary, out/'summary.json')
        if set(QA).issubset(summary):
            from .qa_eval_settings import write_qa_results
            write_qa_results({name: summary[name] for name in QA}, out, predictions_folder='.')
        print(json.dumps(summary, indent=2))
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
