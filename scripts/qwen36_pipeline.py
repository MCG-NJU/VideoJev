"""Immutable source/data snapshots and completion audits for the 8-node MoE run."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil

from scripts.pipeline import read, digest, save, verify_completion

PROFILE = 'qwen36_mixed_llm_lr2e6_gbs64_8nodes_v1'


def snapshot(project, out, data):
    out.mkdir(parents=True, exist_ok=True)
    target = out/'source'
    with (out/'.snapshot.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (target/'snapshot.json').exists():
            state = read(target/'snapshot.json')
            if state['profile'] != PROFILE:
                raise ValueError('Run directory belongs to another recipe')
            for name, value in state['files'].items():
                if digest(out/name) != value:
                    raise ValueError(f'Immutable run file changed: {name}')
            return target
        if target.exists() or (out/'data').exists():
            raise ValueError('Incomplete snapshot; inspect it or choose a new WORK_DIR')
        ready = read(data/'ready.json')
        if not ready.get('complete') or sum(s['questions'] for s in ready['summary'].values()) != 404037:
            raise ValueError('Run prepare_qwen36_mixed.py before launch')
        for name, value in ready['files'].items():
            if digest(data/name) != value:
                raise ValueError(f'Prepared data changed: {name}')
        staging = out/f'.snapshot.{os.getpid()}'
        staging.mkdir()
        source = staging/'source'
        source.mkdir()
        for name in ('timejev', 'scripts', 'configs', *(['videojev_hf'] if (project/'videojev_hf').is_dir() else [])):
            shutil.copytree(project/name, source/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for name in ('sitecustomize.py', 'requirements-overlay.txt'):
            shutil.copy2(project/name, source/name)
        for name in ('.deps', '.deps-qwen36'):
            if (project/name).exists():
                (source/name).symlink_to((project/name).resolve(), target_is_directory=True)
        for config_file in (source/'configs').glob('*.json'):
            save(read(config_file), config_file)
        (staging/'data').mkdir()
        for name in (*ready['files'], 'ready.json'):
            destination = staging/'data'/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(data/name, destination)
        for stage in ('head_only', 'joint'):
            config = read(source/f'configs/qwen36_{stage}.json')
            config.update(deepspeed=str(target/'configs/deepspeed_qwen36_zero3.json'),
                          qa_evaluation_config=str(target/'configs/qa_evaluation.json'))
            for key in ('train', 'packed_train_lengths', 'packed_train_plan', 'training_split_report',
                        'rgb_metadata', 'rgb_evaluation', 'rgb_ready'):
                relative = Path(config[key]).relative_to(project/'data/qwen36_mixed_v1')
                config[key] = str(out/'data'/relative)
            save(config, source/f'configs/qwen36_{stage}.json')
        files = {str(p.relative_to(staging)): digest(p) for folder in ('source', 'data')
                 for p in sorted((staging/folder).rglob('*')) if p.is_file()}
        save(dict(profile=PROFILE, project=str(project), files=files), source/'snapshot.json')
        (staging/'data').rename(out/'data')
        source.rename(target)
        staging.rmdir()
    return target


def stage_action(out, stage):
    directory = out/stage
    marker = directory/('head_stage_complete.json' if stage == 'head_only' else 'completion.json')
    if not marker.exists() or not read(marker).get('complete'):
        return 'run'
    if stage == 'head_only':
        if not (directory/'head.pt').is_file():
            raise ValueError('Completed head export is missing')
    else:
        export = directory/'hf_checkpoint'
        state = read(export/'complete_manifest.json')
        if not state.get('complete'):
            raise ValueError('Joint export incomplete')
        for name, size in state['files'].items():
            if not (export/name).is_file() or (export/name).stat().st_size != size:
                raise ValueError(f'Missing/incomplete joint export shard: {name}')
    return 'skip'


def finalize(out):
    evaluation = verify_completion(out/'evaluation')
    if read(out/'joint/completion.json')['checkpoint_id'] != evaluation['identity']['checkpoint_id']:
        raise ValueError('Evaluated checkpoint differs from the completed joint training run')
    expected = {'charades', 'activitynet', 'qvhighlights'} | {
        f'{name}_{capacity}' for name in ('lvbench', 'videomme', 'longvideobench', 'mlvu')
        for capacity in (196608, 262144)}
    if set(evaluation['benchmarks']) != expected:
        raise ValueError('All eleven results are required')
    for folder in ('grounding', 'qa_196608', 'qa_262144'):
        names = ('charades', 'activitynet', 'qvhighlights') if folder == 'grounding' else (
            'lvbench', 'videomme', 'longvideobench', 'mlvu')
        for name in names:
            state = verify_completion(out/'evaluation'/folder/name)
            if state['identity']['checkpoint_sha256'] != evaluation['identity']['checkpoint_sha256']:
                raise ValueError('Benchmarks must use the same checkpoint')
    audits = {}
    for stage in ('head_only', 'joint'):
        if stage_action(out, stage) != 'skip':
            raise ValueError('Both stages must have complete exports')
        directory = out/stage
        config = read(directory/'training_config.json')
        steps, questions = config['total_steps'], config['unique_training_queries']
        if config['world_size'] != 64:
            raise ValueError('Expected a 64-GPU training run')
        for rank in range(64):
            rows = [json.loads(line) for line in (directory/f'train_rank{rank}.jsonl').read_text().splitlines()]
            if ([r['step'] for r in rows] != list(range(1, steps+1))
                    or sum(r['global_samples'] for r in rows) != questions
                    or any(r['vtg_queries']+r['qa_queries'] != r['global_samples'] for r in rows)):
                raise ValueError('Training logs do not cover all planned questions exactly once')
        audits[stage] = dict(steps=steps, questions=questions, ranks=64)
    scores = read(out/'evaluation/summary.json')
    from timejev.qa_eval_settings import select_context_results
    qa = ('lvbench', 'videomme', 'longvideobench', 'mlvu')
    selected = select_context_results({str(c): {n: scores[f'{n}_{c}'] for n in qa}
                                       for c in (196608, 262144)})
    lines = ['# VideoJev-35B-A3B results', '',
             f"Checkpoint: `{evaluation['identity']['checkpoint_id']}`", '',
             '| Benchmark | Score (%) | Questions |', '|---|---:|---:|']
    for name in ('charades', 'activitynet', 'qvhighlights', *qa):
        metrics = scores[name] if name not in qa else selected[name]['metrics']
        score = metrics['mIoU'] if name not in qa else 100*metrics[selected[name]['metric']]
        lines.append(f"| {name} | {score:.4f} | {metrics['n']} |")
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    save(dict(complete=True, checkpoint_id=evaluation['identity']['checkpoint_id'],
              training=audits, benchmarks=sorted(expected)), out/'completion.json')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=('snapshot', 'action', 'finalize'))
    p.add_argument('--project', default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--out', required=True)
    p.add_argument('--data')
    p.add_argument('--stage', choices=('head_only', 'joint'))
    a = p.parse_args()
    out, project = Path(a.out).resolve(), Path(a.project).resolve()
    if a.operation == 'snapshot':
        print(snapshot(project, out, Path(a.data).resolve() if a.data else project/'data/qwen36_mixed_v1'))
    elif a.operation == 'action':
        print(stage_action(out, a.stage))
    else:
        finalize(out)
