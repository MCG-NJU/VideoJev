"""Small, metadata-only lifecycle helper for the eight-node pipeline."""
from timejev.config import read_config
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

BENCHMARKS = {'charades', 'activitynet', 'qvhighlights', 'lvbench', 'videomme', 'longvideobench', 'mlvu'}


def read(path):
    return read_config(path)


def digest(path):
    if Path(path).is_dir():
        from timejev.hf_checkpoint import checkpoint_fingerprint
        return checkpoint_fingerprint(path)
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8*1024*1024), b''):
            h.update(part)
    return h.hexdigest()


def save(value, path):
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def snapshot(project, out, profile=None):
    out.mkdir(parents=True, exist_ok=True)
    target = out/'source'
    with (out/'.snapshot.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (target/'snapshot.json').exists():
            state = read(target/'snapshot.json')
            if state.get('profile') != profile:
                raise ValueError('Run directory belongs to another input/training profile')
            if digest(Path(state['project'])/'data/ready.json') != state['data_ready_sha256']:
                raise ValueError('Prepared data identity changed since this run was created')
            for name, value in state['files'].items():
                if digest(target/name) != value:
                    raise ValueError(f'Run source snapshot changed: {name}')
            return target
        if target.exists():
            raise ValueError('Incomplete source snapshot; inspect it before reuse')
        if not (project/'data/ready.json').exists():
            raise ValueError('Prepare the data bundle documented in docs/DATA.md before launching')
        staging = out/f'.source.{os.getpid()}'
        staging.mkdir()
        for name in ('timejev', 'scripts', 'configs', *(['videojev_hf'] if (project/'videojev_hf').is_dir() else [])):
            shutil.copytree(project/name, staging/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for name in ('sitecustomize.py', 'requirements-overlay.txt'):
            shutil.copy2(project/name, staging/name)
        if (project/'.deps').exists():
            (staging/'.deps').symlink_to((project/'.deps').resolve(), target_is_directory=True)
        for config_file in (staging/'configs').glob('*.json'):
            save(read(config_file), config_file)
        for stage in ('head_only', 'joint'):
            config = read(staging/f'configs/{"rgb192_" if profile == "rgb192" else ""}{stage}.json')
            config.update(deepspeed=str(target/f'configs/deepspeed_zero{3 if profile == "rgb192" else 2}.json'),
                          qa_evaluation_config=str(target/'configs/qa_evaluation.json'))
            if profile == 'rgb192':
                for key in ('train', 'packed_train_lengths', 'packed_train_plan', 'training_split_report',
                            'rgb_metadata', 'rgb_evaluation', 'rgb_ready'):
                    relative = Path(config[key]).relative_to(project/'data/rgb192')
                    config[key] = str(out/'data'/relative)
            save(config, staging/f'configs/{stage}.json')
        files = {str(p.relative_to(staging)): digest(p) for folder in ('timejev', 'scripts', 'configs', *(['videojev_hf'] if (project/'videojev_hf').is_dir() else []))
                 for p in sorted((staging/folder).rglob('*')) if p.is_file()}
        files.update({name: digest(staging/name) for name in ('sitecustomize.py', 'requirements-overlay.txt')})
        save(dict(files=files, project=str(project), profile=profile, data_ready_sha256=digest(project/'data/ready.json')),
             staging/'snapshot.json')
        staging.rename(target)
    return target


def action(out, stage):
    directory = out/stage
    marker = directory/'completion.json'
    artifact = directory/('head.pt' if stage == 'head_only' else 'hf_checkpoint')
    if stage != 'head_only' and not artifact.exists():
        artifact = directory/'model.pt'  # Read compatibility for previous runs.
    valid = artifact.is_file() or (artifact/'export_complete.json').is_file()
    if marker.exists() and read(marker).get('complete') and valid:
        return 'skip'
    if not (directory/'training_config.json').exists():
        return 'start'
    candidates = sorted((directory/'safety_deepspeed').glob('step*/complete_manifest.json'))
    if not candidates:
        raise ValueError(f'No complete safety checkpoint in {directory}; keep this attempt and use a new WORK_DIR')
    state = read(candidates[-1])
    if not state.get('complete') or state['world_size'] != 64:
        raise ValueError('Resume requires a complete 64-rank safety checkpoint')
    for name, size in state['files'].items():
        path = candidates[-1].parent/name
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f'Incomplete resume file: {path}')
    return 'resume'


def verify_completion(directory):
    state = read(directory/'completion.json')
    if not state.get('complete') or not state.get('files'):
        raise ValueError(f'Incomplete evaluation: {directory}')
    for name, value in state['files'].items():
        path = directory/name
        if not path.is_file() or digest(path) != value:
            raise ValueError(f'Evaluation output changed: {path}')
    return state


def audit_training(directory):
    config = read(directory/'training_config.json')
    completion = read(directory/'completion.json')
    steps, questions = config['total_steps'], config['unique_training_queries']
    if config['world_size'] != 64 or not completion.get('complete') or completion['steps'] != steps:
        raise ValueError('Training completion differs from the planned 64-rank run')
    for rank in range(64):
        rows = [json.loads(line) for line in (directory/f'train_rank{rank}.jsonl').read_text().splitlines()]
        if ([r['step'] for r in rows] != list(range(1, steps+1))
                or sum(r['global_samples'] for r in rows) != questions
                or any(r['vtg_queries']+r['qa_queries'] != r['global_samples'] for r in rows)):
            raise ValueError(f'Incomplete question/update coverage in rank {rank}: {directory}')
    return dict(steps=steps, questions=questions, ranks=64, complete=True)


def cleanup(out, evaluation_dir=None):
    evaluation_dir = out/'evaluation' if evaluation_dir is None else Path(evaluation_dir)
    evaluation = verify_completion(evaluation_dir)
    if set(evaluation['benchmarks']) != BENCHMARKS:
        raise ValueError('All seven benchmark results are required before cleanup')
    for name in BENCHMARKS:
        state = verify_completion(evaluation_dir/name)
        if state['identity']['checkpoint_id'] != evaluation['identity']['checkpoint_id']:
            raise ValueError('Benchmarks use different checkpoints')
    training = {}
    for stage in ('head_only', 'joint'):
        if action(out, stage) != 'skip':
            raise ValueError('Both completed stage exports are required before cleanup')
        training[stage] = audit_training(out/stage)
    exported = out/'joint/hf_checkpoint'
    if not exported.exists():
        exported = out/'joint/model.pt'
    if digest(exported) != evaluation['identity']['checkpoint_sha256']:
        raise ValueError('Evaluated checkpoint differs from the model export')
    removed = []
    for stage in ('head_only', 'joint'):
        directory = out/stage
        candidates = [directory/'safety_deepspeed', *directory.glob('resume_archives/*/safety_deepspeed')]
        for path in candidates:
            if path.exists():
                if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
                    raise ValueError('Refuse cleanup outside this pipeline')
                shutil.rmtree(path)
                removed.append(str(path))
    save(dict(complete=True, removed=removed, training_audit=training, checkpoint_id=evaluation['identity']['checkpoint_id'],
              evaluation_dir=str(evaluation_dir),
              retained=['head_only/head.pt', str(exported.relative_to(out)), 'source', str(evaluation_dir), 'training logs']),
         out/'cleanup.json')
    summary = read(evaluation_dir/'summary.json')
    lines = ['# Video-Jev results', '', f"Checkpoint: `{evaluation['identity']['checkpoint_id']}`", '',
             '| Benchmark | Primary score (%) | Questions |', '|---|---:|---:|']
    for name in ('charades', 'activitynet', 'qvhighlights', 'lvbench', 'videomme', 'longvideobench', 'mlvu'):
        metrics = summary[name]
        score = metrics['mIoU'] if name in ('charades','activitynet','qvhighlights') else 100*metrics['m_avg' if name == 'mlvu' else 'accuracy']
        lines.append(f"| {name} | {score:.4f} | {metrics['n']} |")
    (out/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    save(dict(complete=True, checkpoint_id=evaluation['identity']['checkpoint_id'], benchmarks=sorted(BENCHMARKS),
              evaluation_dir=str(evaluation_dir)),
         out/'completion.json')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation', choices=('snapshot', 'action', 'cleanup'))
    p.add_argument('--out', required=True)
    p.add_argument('--project', default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--stage', choices=('head_only', 'joint'))
    p.add_argument('--profile', choices=('rgb192',))
    p.add_argument('--evaluation-dir', help='Completed evaluation to verify before cleanup')
    a = p.parse_args()
    out = Path(a.out).resolve()
    if a.operation == 'snapshot':
        print(snapshot(Path(a.project).resolve(), out, a.profile))
    elif a.operation == 'action':
        if a.stage is None:
            p.error('--stage is required for action')
        print(action(out, a.stage))
    else:
        cleanup(out, a.evaluation_dir)


if __name__ == '__main__':
    main()
