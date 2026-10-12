"""Prepare exact 192K RGB geometry, conversation splits and packs; no feature cache."""
from timejev.config import read_config
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import json
import math
import multiprocessing
from pathlib import Path
import time

from timejev.data import read_jsonl
from timejev.engine import save_json, sha256
from timejev.multiturn_inputs import group_memberships
from timejev.rgb_data import CAPACITY, PROTOCOL, geometry, plan_conversation, text_lengths
from timejev.rgb_media import metadata

_WORKER_PROCESSOR = None


def init_worker(base):
    from timejev.decision_inputs import load_processor
    import torch
    torch.set_num_threads(1)
    global _WORKER_PROCESSOR
    _WORKER_PROCESSOR = load_processor(base)


def plan_video_job(job):
    members, benchmark = job
    key = members[0]['video_key']
    try:
        record = metadata(members[0], benchmark)
        first, cont = text_lengths(_WORKER_PROCESSOR, record, members)
        return dict(video_key=key, record=record, first=first, continuation=cont,
                    query_ids=[r['id'] for r in members])
    except Exception as error:
        return dict(video_key=key, error=f'{type(error).__name__}: {error}')


def write_rows(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
    temp.replace(path)


def source_rows(project, seed):
    stages = {s: read_jsonl(project/f'data/{s}/train.jsonl') for s in ('head_only', 'joint')}
    all_rows = [r for rows in stages.values() for r in rows]
    if not all(stages.values()) or len({r['id'] for r in all_rows}) != len(all_rows):
        raise ValueError('Both stages require nonempty manifests with unique question IDs')
    if ({r['video_key'] for r in stages['head_only']} & {r['video_key'] for r in stages['joint']}):
        raise ValueError('A source video crossed training stages')
    paths = [project/f'data/{s}/train.jsonl' for s in stages]
    return stages, {str(p): sha256(p) for p in paths}


def plan_records(rows, processor, output, *, benchmark=None, workers=12):
    """Incremental JSON-only planning; extraction-in-progress failures are retriable."""
    episodes = group_memberships(rows)
    saved = {}
    if output.exists():
        for line in output.read_text().splitlines():
            value = json.loads(line)
            saved[value['video_key']] = value
    errors, done, start = [], {}, time.monotonic()
    missing = []
    for episode in episodes:
        key = episode['video_key']
        members = [rows[i] for i in episode['query_indices']]
        if key in saved:
            value = saved[key]
            if value['query_ids'] != [r['id'] for r in members]:
                raise ValueError('Incremental RGB plan question identity changed')
            done[key] = value
        else:
            missing.append((members, benchmark))
    if not missing:
        return episodes, done
    with output.open('a', buffering=1) as journal, ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context('spawn'), initializer=init_worker,
            initargs=(processor.tokenizer.name_or_path,)) as pool:
        for index, value in enumerate(pool.map(plan_video_job, missing, chunksize=8), len(done)+1):
            if 'error' in value:
                errors.append(value)
            else:
                done[value['video_key']] = value
                if value['video_key'] not in saved:
                    journal.write(json.dumps(value, ensure_ascii=False)+'\n')
            if index % 1000 == 0 or index == len(episodes):
                print(json.dumps(dict(event='rgb_metadata', benchmark=benchmark or 'train',
                    complete=index, total=len(episodes), unavailable=len(errors),
                    seconds=round(time.monotonic()-start, 1))), flush=True)
    save_json(errors, output.with_suffix('.unavailable.json'))
    if errors:
        raise FileNotFoundError(f'{len(errors)} RGB sources unavailable/invalid; see '
                                f'{output.with_suffix(".unavailable.json")}. Fix/extract them and rerun.')
    return episodes, done


def prepare(project, root, base, workers, *, benchmarks=None):
    from timejev.decision_inputs import load_processor
    from timejev.packing import build_xtuner_packs, xtuner_source_hashes
    from timejev.token_metadata import tokenizer_identity
    if benchmarks is None:
        benchmarks = ('charades', 'activitynet', 'qvhighlights', 'lvbench', 'videomme', 'longvideobench', 'mlvu')
    request = dict(base=str(Path(base).resolve()), benchmarks=list(benchmarks), qa_context=262144)
    root.mkdir(parents=True, exist_ok=True)
    with (root/'.prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        stages, bindings = source_rows(project, 20260924)
        bindings.update({str(project/f'data/benchmarks/{name}/questions.jsonl'):
                         sha256(project/f'data/benchmarks/{name}/questions.jsonl') for name in benchmarks})
        if (root/'ready.json').exists():
            ready = json.loads((root/'ready.json').read_text())
            if ready['source_bindings'] != bindings or ready.get('preparation') != request:
                raise ValueError('RGB inputs/settings changed; prepare a new data directory')
            for name, digest in ready['files'].items():
                if sha256(root/name) != digest:
                    raise ValueError(f'Prepared RGB file changed: {name}')
            print(json.dumps(dict(reused=True, summary=ready['summary'])), flush=True)
            return
        processor = load_processor(base)
        identity = dict(protocol=PROTOCOL, capacity=CAPACITY, preparation=request, source_bindings=bindings,
                        tokenizer=tokenizer_identity(processor, prompt_version=PROTOCOL),
                        code={name: sha256(Path(__file__).resolve().parents[1]/name) for name in
                              ('timejev/rgb_data.py', 'timejev/rgb_media.py', 'timejev/multiturn_inputs.py',
                               'timejev/timestamp_option_inputs.py', 'scripts/prepare_rgb192.py')})
        identity_path = root/'identity.json'
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError('RGB planning identity changed; choose a fresh data/run directory')
        save_json(identity, identity_path)
        all_rows = [r for rows in stages.values() for r in rows]
        _, planned = plan_records(all_rows, processor, root/'train_progress.jsonl', workers=workers)
        records = {key: value['record'] for key, value in planned.items()}
        save_json(dict(protocol=PROTOCOL, records=records), root/'rgb_metadata.json')
        summaries, exclusions, splits = {}, [], []
        for stage, original in stages.items():
            target = root/stage
            target.mkdir(exist_ok=True)
            retained, episodes = [], []
            for group in group_memberships(original):
                value = planned[group['video_key']]
                pieces, dropped = plan_conversation(value['record'], value['first'], value['continuation'])
                exclusions.extend(dict(id=value['query_ids'][i], video_key=group['video_key'], stage=stage,
                                       reason='single_question_exceeds_192k_at_320_squared_pixels') for i in dropped)
                if len(pieces) > 1:
                    splits.append(dict(video_key=group['video_key'], stage=stage, pieces=len(pieces),
                                       query_ids=[[value['query_ids'][i] for i in p['members']] for p in pieces]))
                for part, piece in enumerate(pieces):
                    members = [original[group['query_indices'][i]] for i in piece.pop('members')]
                    indices = list(range(len(retained), len(retained)+len(members)))
                    retained.extend(members)
                    episodes.append(dict(id=f'{group["id"]}/part{part}', video_key=group['video_key'],
                                         query_indices=indices, **piece))
            write_rows(retained, target/'train.jsonl')
            spec = dict(protocol=PROTOCOL, training_layout='shared_video_multiturn_v1',
                        manifest_sha256=sha256(target/'train.jsonl'),
                        rgb_metadata_sha256=sha256(root/'rgb_metadata.json'), max_sequence_length=CAPACITY,
                        seed=20260924, fps=2., episodes=episodes)
            save_json(spec, target/'episodes.json')
            packs = build_xtuner_packs([e['tokens'] for e in episodes], CAPACITY, 20260924)
            save_json(dict(packs=packs, max_sequence_length=CAPACITY, seed=20260924,
                           xtuner_source_sha256=xtuner_source_hashes(),
                           length_manifest_sha256=sha256(target/'episodes.json')), target/'packs.json')
            summaries[stage] = dict(input_questions=len(original), questions=len(retained),
                videos=len({r['video_key'] for r in retained}), episodes=len(episodes), packs=len(packs),
                steps=math.ceil(len(packs)/64), dropped_questions=len(original)-len(retained))
        save_json(dict(protocol=PROTOCOL, seed=20260924, questions_per_epoch=1,
                       existing_video_stages='unchanged', source_questions=len(all_rows),
                       stages=summaries), root/'split.json')
        save_json(dict(dropped_questions=exclusions, split_videos=splits), root/'filter_audit.json')
        evaluation = {}
        from timejev.qa_eval_settings import QA_CONTEXT
        for name in benchmarks:
            context = QA_CONTEXT if name in ('lvbench', 'videomme', 'longvideobench', 'mlvu') else CAPACITY
            manifest = project/f'data/benchmarks/{name}/questions.jsonl'
            rows = read_jsonl(manifest)
            groups, values = plan_records(rows, processor, root/f'{name}_progress.jsonl',
                                         benchmark=name, workers=workers)
            entries = []
            for group in groups:
                value = values[group['video_key']]
                # Use the longest independent question to select one geometry,
                # so every question can privately fork the SAME video prefix.
                if context == QA_CONTEXT:
                    from timejev.context_eval import choose_geometry
                    shape = choose_geometry(value['record'], max(value['first']), context)
                else:
                    shape = geometry(value['record'], max(value['first']), minimum=1024)
                if shape is None:
                    raise ValueError(f'Benchmark single question cannot fit at native minimum: {group["video_key"]}')
                entries.append(dict(video_key=group['video_key'], record=value['record'], geometry=shape,
                    questions=[dict(id=rows[i]['id'], query_index=i, tokens=first+shape['tokens'])
                               for i, first in zip(group['query_indices'], value['first'])]))
            evaluation[name] = dict(manifest=str(manifest), manifest_sha256=sha256(manifest),
                                    context=context, videos=entries)
        save_json(dict(protocol=PROTOCOL, max_sequence_length=CAPACITY, benchmarks=evaluation), root/'evaluation.json')
        files = {str(p.relative_to(root)): sha256(p) for p in root.rglob('*.json')
                 if not p.name.endswith('.unavailable.json')}
        files.update({f'{s}/train.jsonl': sha256(root/s/'train.jsonl') for s in stages})
        save_json(dict(complete=True, preparation=request, source_bindings=bindings, files=files, summary=summaries), root/'ready.json')
        print(json.dumps(dict(event='rgb_preparation_complete', summary=summaries)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument('--out', required=True)
    parser.add_argument('--base', default=read_config(Path(__file__).resolve().parents[1]/'configs/head_only.json')['base'])
    parser.add_argument('--workers', type=int, default=12)
    args = parser.parse_args()
    prepare(Path(args.project), Path(args.out), args.base, args.workers)


if __name__ == '__main__':
    main()
