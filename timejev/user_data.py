"""Portable JSONL manifests for training and evaluation."""
import json
import math
import random
from pathlib import Path

from .decision_inputs import validate_row
from .multiturn_inputs import group_memberships


def load_manifest(path):
    """Resolve media paths relative to the manifest and validate labeled rows."""
    path = Path(path).resolve()
    rows, owners = [], {}
    for number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            for key in ('id', 'video_key', 'video_path', 'query'):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f'{key} must be a nonempty string')
            row['task'] = row.get('task', 'grounding')
            if isinstance(row.get('duration'), bool):
                raise ValueError('duration must be positive seconds')
            row['duration'] = float(row['duration'])
            if not math.isfinite(row['duration']) or row['duration'] <= 0:
                raise ValueError('duration must be positive seconds')
            validate_row(row, require_labels=True)
            row['video_path'] = str((path.parent / row['video_path']).resolve())
            if 'media_source' in row:
                source = dict(row['media_source'])
                if source['kind'] not in ('video', 'frames_sidecar', 'frames_4fps'):
                    raise ValueError('Unknown media_source kind')
                source['path'] = str((path.parent / source['path']).resolve())
                row['media_source'] = source
            media = row.get('media_source', {'path': row['video_path']})['path']
            if not Path(media).exists():
                raise FileNotFoundError(media)
            owner = owners.setdefault(row['video_path'], row['video_key'])
            if owner != row['video_key']:
                raise ValueError('Use the same video_key for every question about a video')
            rows.append(row)
        except (ValueError, KeyError, TypeError, OSError) as error:
            raise ValueError(f'{path}:{number}: {error}') from error
    group_memberships(rows)  # Unique IDs and consistent video paths/durations.
    sources = {}
    for row in rows:
        source = row.get('media_source')
        if row['video_key'] in sources and sources[row['video_key']] != source:
            raise ValueError(f'Conflicting media_source for {row["video_key"]}')
        sources[row['video_key']] = source
    return rows


def split_stages(rows, head_fraction=.3, seed=20260924):
    """Keep all questions from a video in one stage and preserve row order."""
    if not 0 < head_fraction < 1:
        raise ValueError('head_fraction must be between 0 and 1')
    videos = sorted({row['video_key'] for row in rows})
    if len(videos) < 2:
        raise ValueError('Two-stage training requires at least two videos')
    random.Random(seed).shuffle(videos)
    count = min(len(videos)-1, max(1, round(len(videos)*head_fraction)))
    head = set(videos[:count])
    return {stage: [r for r in rows if (r['video_key'] in head) == (stage == 'head_only')]
            for stage in ('head_only', 'joint')}


def write_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n' for row in rows))
