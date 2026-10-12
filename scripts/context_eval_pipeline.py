"""Immutable source/plan snapshot for the seven-node evaluation-only launcher."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil

from scripts.pipeline import digest, read, save


def snapshot(project, out, plans, checkpoint):
    out.mkdir(parents=True, exist_ok=True)
    target = out/'source'
    with (out/'.snapshot.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        stat = checkpoint.stat()
        request = dict(project=str(project), checkpoint=str(checkpoint),
                       checkpoint_bytes=stat.st_size, checkpoint_mtime_ns=stat.st_mtime_ns)
        if (target/'snapshot.json').exists():
            state = read(target/'snapshot.json')
            if state['request'] != request:
                raise ValueError('Run checkpoint/project changed; use a fresh WORK_DIR')
            for name, value in state['files'].items():
                if digest(target/name) != value:
                    raise ValueError(f'Run source/plan snapshot changed: {name}')
            return target
        if target.exists():
            raise ValueError('Incomplete snapshot exists; inspect it or choose a fresh WORK_DIR')
        ready = read(plans/'ready.json')
        if not ready.get('complete'):
            raise ValueError('Prepare RGB context evaluation plans first')
        for name, value in ready['files'].items():
            if digest(plans/name) != value:
                raise ValueError(f'Prepared evaluation plan changed: {name}')
        identity = read(plans/'identity.json')
        for name, value in identity['code'].items():
            if digest(project/name) != value:
                raise ValueError(f'Planning code changed: {name}; prepare a fresh plan directory')
        staging = out/f'.source.{os.getpid()}'
        staging.mkdir()
        try:
            for name in ('timejev', 'scripts', 'configs', *(['videojev_hf'] if (project/'videojev_hf').is_dir() else [])):
                shutil.copytree(project/name, staging/name, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            for name in ('sitecustomize.py', 'requirements-overlay.txt'):
                shutil.copy2(project/name, staging/name)
            if (project/'.deps').exists():
                (staging/'.deps').symlink_to((project/'.deps').resolve(), target_is_directory=True)
            config_file = staging/'configs/qa_evaluation.json'
            if config_file.exists():
                save(read(config_file), config_file)
            (staging/'plans').mkdir()
            for name in (*ready['files'], 'ready.json'):
                shutil.copy2(plans/name, staging/'plans'/name)
            for name, value in identity['code'].items():
                if digest(staging/name) != value:
                    raise ValueError(f'Planning code changed during snapshot: {name}')
            files = {str(p.relative_to(staging)): digest(p) for folder in ('timejev', 'scripts', 'configs', 'plans', *(['videojev_hf'] if (project/'videojev_hf').is_dir() else []))
                     for p in sorted((staging/folder).rglob('*')) if p.is_file()}
            files.update({name: digest(staging/name) for name in ('sitecustomize.py', 'requirements-overlay.txt')})
            save(dict(request=request, files=files), staging/'snapshot.json')
            staging.rename(target)
        except BaseException:
            shutil.rmtree(staging)
            raise
    return target


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('project', 'out', 'plans', 'checkpoint'):
        p.add_argument('--'+name, required=True)
    a = p.parse_args()
    print(snapshot(*(Path(getattr(a, name)).resolve() for name in ('project', 'out', 'plans', 'checkpoint'))))
