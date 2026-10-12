"""Prepare fixed-stage complete conversations and 144K training/grounding and 256K QA RGB plans."""
from timejev.config import read_config
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import math
import multiprocessing
from pathlib import Path
import shutil

from timejev.data import read_jsonl
from timejev.engine import sha256, save_json
from timejev.qwen36_data import PROTOCOL, CAPACITY

PROCESSOR = None


def init_worker(base):
    from timejev.decision_inputs import load_processor
    import torch
    torch.set_num_threads(1)
    global PROCESSOR
    PROCESSOR = load_processor(base)


def job(args):
    from timejev.rgb_data import text_lengths
    from timejev.rgb_media import metadata, fingerprint
    from timejev.context_eval import full_frame_record
    from timejev.vision import resolution_for_budget
    members, benchmark = args
    row = members[0]
    if row.get('task') != 'choice':
        record = metadata(row, benchmark)
    else:
        record = full_frame_record(row, benchmark) if benchmark else metadata(row)
    first, continuation = text_lengths(PROCESSOR, record, members)
    shape = resolution_for_budget(PROCESSOR, record['frames'], *record['original_hw'],
                                  visual_token_budget=CAPACITY*4//5)
    h, w = shape['processed_hw']
    geometry = dict(processed_hw=[h, w], grid_thw=[(record['frames']+1)//2, h//16, w//16],
                    tokens=shape['visual_tokens'], pixels_per_frame=h*w)
    return dict(video_key=row['video_key'], record=record, first=first, continuation=continuation,
                geometry=geometry, query_ids=[r['id'] for r in members])


def records(pool, jobs, journal_path):
    saved = {}
    if journal_path.exists():
        for line in journal_path.read_text().splitlines():
            item = json.loads(line)
            saved[item['video_key']] = item
    pending = []
    for members, benchmark in jobs:
        key = members[0]['video_key']
        if key in saved:
            if saved[key]['query_ids'] != [r['id'] for r in members]:
                raise ValueError('Conversation ordering changed')
        else:
            pending.append((members, benchmark))
    with journal_path.open('a', buffering=1) as journal:
        for item in pool.map(job, pending, chunksize=16):
            journal.write(json.dumps(item, ensure_ascii=False)+'\n')
            saved[item['video_key']] = item
            if len(saved) % 1000 == 0:
                print(json.dumps(dict(event='planned_videos', journal=journal_path.name,
                                      done=len(saved), total=len(jobs))), flush=True)
    return saved


def prepare(project, root, base, workers, *, benchmarks=None):
    from timejev.decision_inputs import load_processor
    from timejev.token_metadata import tokenizer_identity
    from timejev.packing import build_xtuner_packs, xtuner_source_hashes
    from timejev.multiturn_inputs import group_memberships
    from timejev.context_eval import CONTEXTS, COUNTS, choose_geometry, validate_spec
    root.mkdir(parents=True, exist_ok=True)
    with (root/'.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        from scripts.prepare_rgb192 import source_rows
        stage_rows, _ = source_rows(project, 20260924)
        stages = tuple(stage_rows)
        if benchmarks is None:
            benchmarks = ('charades', 'activitynet', 'qvhighlights', *COUNTS)
        paths = [project/f'data/{s}/train.jsonl' for s in stages]
        paths += [project/'data/split.json']
        paths += [project/f'data/benchmarks/{n}/questions.jsonl'
                  for n in benchmarks]
        code_root = Path(__file__).resolve().parents[1]
        identity = dict(protocol=PROTOCOL, base=base, capacity=CAPACITY, sample_visual_fraction=.8,
            source_bindings={str(p): sha256(p) for p in paths},
            tokenizer=tokenizer_identity(load_processor(base), prompt_version=PROTOCOL),
            code={n: sha256(code_root/n) for n in ('scripts/prepare_qwen36_mixed.py',
                'timejev/rgb_data.py', 'timejev/rgb_media.py', 'timejev/multiturn_inputs.py',
                'timejev/timestamp_option_inputs.py', 'timejev/vision.py', 'timejev/context_eval.py')})
        if (root/'identity.json').exists() and json.loads((root/'identity.json').read_text()) != identity:
            raise ValueError('Mixed preparation identity changed; use a new data directory')
        if (root/'ready.json').exists():
            ready = json.loads((root/'ready.json').read_text())
            for n, digest in ready['files'].items():
                if sha256(root/n) != digest:
                    raise ValueError(f'Prepared file changed: {n}')
            print(json.dumps(dict(event='prepared_data_reused', summary=ready['summary'])), flush=True)
            return
        save_json(identity, root/'identity.json')
        summaries, catalog, overlength = {}, {}, []
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=init_worker, initargs=(base,)) as pool:
            for stage in stages:
                rows = read_jsonl(project/f'data/{stage}/train.jsonl')
                parent = dict(episodes=group_memberships(rows))
                jobs = [([rows[i] for i in g['query_indices']], None) for g in parent['episodes']]
                values = records(pool, jobs, root/f'{stage}_progress.jsonl')
                target = root/stage
                target.mkdir(exist_ok=True)
                shutil.copy2(project/f'data/{stage}/train.jsonl', target/'train.jsonl')
                episodes = []
                for original in parent['episodes']:
                    value = values[original['video_key']]
                    tokens = value['first'][0]+sum(value['continuation'][1:])+value['geometry']['tokens']
                    if tokens > CAPACITY:
                        overlength.append(dict(stage=stage, video_key=original['video_key'],
                            tokens=tokens, questions=len(original['query_indices'])))
                    episodes.append(dict(original, geometry=value['geometry'], tokens=tokens))
                    catalog[original['video_key']] = value['record']
                save_json(dict(protocol=PROTOCOL, training_layout='shared_video_multiturn_v1',
                    manifest_sha256=sha256(target/'train.jsonl'), max_sequence_length=CAPACITY,
                    seed=20260924, fps=2., episodes=episodes), target/'episodes.json')
                if not overlength:
                    packs = build_xtuner_packs([e['tokens'] for e in episodes], CAPACITY, 20260924)
                    save_json(dict(packs=packs, max_sequence_length=CAPACITY, seed=20260924,
                        xtuner_source_sha256=xtuner_source_hashes(),
                        length_manifest_sha256='pending'), target/'packs.json')
                    summaries[stage] = dict(questions=len(rows), videos=len(episodes), packs=len(packs),
                        steps=math.ceil(len(packs)/64), max_tokens=max(e['tokens'] for e in episodes),
                        vtg_questions=sum(r.get('task') != 'choice' for r in rows),
                        qa_questions=sum(r.get('task') == 'choice' for r in rows))
            save_json(dict(overlength=overlength, split_conversations=0, dropped_questions=0), root/'filter_audit.json')
            if overlength:
                raise ValueError(f'{len(overlength)} complete conversations exceed original 144K recipe; '
                                 'no data dropped or split. See filter_audit.json; user decision required.')
            if sum(s['questions'] for s in summaries.values()) != sum(map(len, stage_rows.values())):
                raise ValueError('Prepared questions differ from the input manifests')
            save_json(dict(protocol=PROTOCOL, records=catalog), root/'rgb_metadata.json')
            for stage in stages:
                path = root/stage/'episodes.json'
                spec = json.loads(path.read_text())
                spec['rgb_metadata_sha256'] = sha256(root/'rgb_metadata.json')
                save_json(spec, path)
                path = root/stage/'packs.json'
                spec = json.loads(path.read_text())
                spec['length_manifest_sha256'] = sha256(root/stage/'episodes.json')
                save_json(spec, path)
            split = json.loads((project/'data/split.json').read_text())
            split.update(protocol=PROTOCOL, stages=summaries, base=base)
            save_json(split, root/'split.json')
            plans = {c: dict(protocol=PROTOCOL, base=base, context=c, visual_budget=c*4//5,
                            identity=identity, benchmarks={}) for c in (CAPACITY, *CONTEXTS)}
            for name in benchmarks:
                path = project/f'data/benchmarks/{name}/questions.jsonl'
                rows = read_jsonl(path)
                groups = group_memberships(rows)
                values = records(pool, [([rows[i] for i in g['query_indices']], name) for g in groups],
                                 root/f'{name}_progress.jsonl')
                for context in CONTEXTS if name in COUNTS else (CAPACITY,):
                    videos = []
                    for group in groups:
                        value = values[group['video_key']]
                        shape = (choose_geometry(value['record'], max(value['first']), context)
                                 if name in COUNTS else value['geometry'])
                        videos.append(dict(video_key=group['video_key'], record=value['record'], geometry=shape,
                            questions=[dict(id=rows[i]['id'], query_index=i, tokens=first+shape['tokens'])
                                       for i, first in zip(group['query_indices'], value['first'])]))
                    spec = dict(manifest=str(path), manifest_sha256=sha256(path), videos=videos)
                    if name in COUNTS:
                        if len(rows) != COUNTS[name]:
                            raise ValueError(f'Incomplete QA inventory: {name}')
                        validate_spec(spec, rows, context)
                    elif any(q['tokens'] > context for v in videos for q in v['questions']):
                        raise ValueError('Grounding question exceeds the original context')
                    plans[context]['benchmarks'][name] = spec
                print(json.dumps(dict(event='evaluation_planned', benchmark=name, questions=len(rows))), flush=True)
            for c, plan in plans.items():
                save_json(plan, root/f'evaluation_{c}.json')
        files = {str(p.relative_to(root)): sha256(p) for p in root.rglob('*.json')}
        files.update({f'{s}/train.jsonl': sha256(root/s/'train.jsonl') for s in stages})
        save_json(dict(complete=True, files=files, summary=summaries), root/'ready.json')
        print(json.dumps(dict(event='mixed_preparation_complete', summary=summaries)), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--out', required=True)
    p.add_argument('--base', default=read_config(Path(__file__).resolve().parents[1]/'configs/qwen36_head_only.json')['base'])
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    prepare(Path(a.project).resolve(), Path(a.out).resolve(), a.base, a.workers)
