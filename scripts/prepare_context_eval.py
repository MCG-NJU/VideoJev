"""Prepare both RGB MCQ contexts, without reading training data or feature caches."""
from timejev.config import read_config
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import multiprocessing
from pathlib import Path

from timejev.context_eval import (CONTEXTS, COUNTS, VIDEO_COUNTS, PROTOCOL, check_source,
                                  full_frame_record, choose_geometry, validate_spec, visual_budget)
from timejev.data import read_jsonl
from timejev.decision_inputs import load_processor
from timejev.engine import save_json, sha256
from timejev.multiturn_inputs import group_memberships
from timejev.rgb_data import text_lengths
from timejev.rgb_media import _frame_paths
from timejev.token_metadata import tokenizer_identity

PROCESSOR = None


def init_worker(base):
    import torch
    torch.set_num_threads(1)
    global PROCESSOR
    PROCESSOR = load_processor(base)


def job(args):
    name, members = args
    record = full_frame_record(members[0], name)
    _frame_paths(record)  # Verify the exact timestamp grid without decoding RGB.
    first, _ = text_lengths(PROCESSOR, record, members)
    return dict(video_key=record['video_key'], record=record, first=first,
                query_ids=[r['id'] for r in members])


def prepare(project, out, base, workers):
    out.mkdir(parents=True, exist_ok=True)
    with (out/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        registry = read_config(project/'configs/qa_evaluation.json')
        refs = {n: {k: v for k, v in registry[n]['2'].items()
                     if k in ('manifest', 'official_questions', 'lvbench_metadata')} for n in COUNTS}
        root = Path(__file__).resolve().parents[1]
        processor = load_processor(base)
        identity = dict(protocol=PROTOCOL, contexts=list(CONTEXTS), visual_budget_fraction=.8,
            sources={p: sha256(p) for ref in refs.values() for p in ref.values()},
            tokenizer=tokenizer_identity(processor, prompt_version=PROTOCOL),
            code={name: sha256(root/name) for name in ('scripts/prepare_context_eval.py',
                'timejev/context_eval.py', 'timejev/rgb_data.py', 'timejev/rgb_media.py',
                'timejev/multiturn_inputs.py', 'timejev/timestamp_option_inputs.py',
                'timejev/token_metadata.py', 'timejev/decision_inputs.py')})
        identity_path = out/'identity.json'
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError('Planning identity changed; use a fresh plan directory')
        save_json(identity, identity_path)
        if (out/'ready.json').exists():
            ready = json.loads((out/'ready.json').read_text())
            for file, digest in ready['files'].items():
                if sha256(out/file) != digest:
                    raise ValueError(f'Prepared plan changed: {file}')
            plan = json.loads((out/f'context_{CONTEXTS[0]}.json').read_text())
            for spec in plan['benchmarks'].values():
                for video in spec['videos']:
                    check_source(video['record'])
            print(json.dumps(dict(event='plans_reused', directory=str(out))), flush=True)
            return
        plans = {c: dict(protocol=PROTOCOL, base=base, context=c, visual_budget=visual_budget(c),
                         identity=identity, benchmarks={}) for c in CONTEXTS}
        summary = {c: {} for c in CONTEXTS}
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=init_worker, initargs=(base,)) as pool:
            for name, ref in refs.items():
                rows = read_jsonl(ref['manifest'])
                groups = group_memberships(rows)
                if (len(rows) != COUNTS[name] or len({r['id'] for r in rows}) != len(rows)
                        or len(groups) != VIDEO_COUNTS[name]):
                    raise ValueError(f'Unexpected benchmark inventory: {name}')
                journal_path = out/f'{name}_progress.jsonl'
                saved = {}
                if journal_path.exists():
                    for line in journal_path.read_text().splitlines():
                        value = json.loads(line)
                        check_source(value['record'])
                        saved[value['video_key']] = value
                missing = [(name, [rows[i] for i in g['query_indices']]) for g in groups
                           if g['video_key'] not in saved]
                with journal_path.open('a', buffering=1) as journal:
                    for value in pool.map(job, missing, chunksize=4):
                        journal.write(json.dumps(value, ensure_ascii=False)+'\n')
                        saved[value['video_key']] = value
                        if len(saved) % 100 == 0:
                            print(json.dumps(dict(event='planning', benchmark=name, videos=len(saved),
                                                  total=len(groups))), flush=True)
                for context, plan in plans.items():
                    videos = []
                    for group in groups:
                        value = saved[group['video_key']]
                        if value['query_ids'] != [rows[i]['id'] for i in group['query_indices']]:
                            raise ValueError('Journal question identity changed')
                        shape = choose_geometry(value['record'], max(value['first']), context)
                        videos.append(dict(video_key=group['video_key'], record=value['record'], geometry=shape,
                            questions=[dict(id=rows[i]['id'], query_index=i, tokens=shape['tokens']+n)
                                       for i, n in zip(group['query_indices'], value['first'])]))
                    spec = dict(**ref, manifest_sha256=sha256(ref['manifest']), videos=videos)
                    validate_spec(spec, rows, context)
                    plan['benchmarks'][name] = spec
                    summary[context][name] = dict(questions=len(rows), videos=len(videos),
                        frames=sum(v['record']['frames'] for v in videos),
                        max_visual_tokens=max(v['geometry']['tokens'] for v in videos),
                        max_input_tokens=max(q['tokens'] for v in videos for q in v['questions']),
                        pixel_area_range=[min(v['geometry']['pixels_per_frame'] for v in videos),
                                          max(v['geometry']['pixels_per_frame'] for v in videos)])
                print(json.dumps(dict(event='planned', benchmark=name, videos=len(groups))), flush=True)
        files = []
        for context, plan in plans.items():
            path = out/f'context_{context}.json'
            save_json(plan, path)
            files.append(path)
        save_json(summary, out/'summary.json')
        files.extend((out/'summary.json', identity_path))
        save_json(dict(complete=True, files={p.name: sha256(p) for p in files}), out/'ready.json')
        print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--out', required=True)
    p.add_argument('--base', default=read_config(Path(__file__).resolve().parents[1]/'configs/head_only.json')['base'])
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    if a.workers < 1:
        p.error('--workers must be positive')
    prepare(Path(a.project).resolve(), Path(a.out).resolve(), a.base, a.workers)
