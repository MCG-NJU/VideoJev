"""Read-only catalogues of independently verified native feature caches."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

LAYOUT = 'multi_root_reference_v1'
COMMON = ('protocol', 'vision_sha256', 'vision_file_sha256', 'requested_fps',
          'max_frames', 'visual_token_budget', 'dimension', 'dtype')


def identity(spec):
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load(root):
    root = Path(root)
    if not (root/'provenance.json').is_file():
        return None
    spec = read(root/'provenance.json')
    if spec.get('cache_layout') != LAYOUT:
        return None
    unsigned = {k: v for k, v in spec.items() if k != 'cache_id'}
    if identity(unsigned) != spec['cache_id']:
        raise ValueError('Feature catalogue provenance identity changed')
    index = read(root/'index.json')
    if (index.get('complete') is not True or index['cache_id'] != spec['cache_id']
            or index['videos'] != len(index['entries'])):
        raise ValueError('Feature catalogue is incomplete or inconsistent')
    if identity(index['entries']) != spec['entries_sha256']:
        raise ValueError('Feature catalogue assignments changed')
    members = {}
    for path, binding in spec['source_caches'].items():
        member = Path(path)
        if member.resolve() == root.resolve():
            raise ValueError('A feature catalogue cannot reference itself')
        if (sha(member/'provenance.json') != binding['provenance_sha256']
                or sha(member/'index.json') != binding['index_sha256']):
            raise ValueError(f'Source feature cache identity/index changed: {member}')
        source, source_index = read(member/'provenance.json'), read(member/'index.json')
        if (source.get('cache_layout') or source['cache_id'] != binding['cache_id']
                or source_index.get('complete') is not True
                or source_index['cache_id'] != source['cache_id']
                or any(source[key] != spec[key] for key in COMMON)):
            raise ValueError(f'Incompatible source feature cache: {member}')
        members[path] = (source, source_index)
    if sorted(spec['member_cache_ids']) != sorted(v[0]['cache_id'] for v in members.values()):
        raise ValueError('Feature catalogue member IDs changed')
    for key, entry in index['entries'].items():
        source, native = members[entry['root']]
        record = native['entries'].get(key)
        if (entry['cache_id'] != source['cache_id'] or record is None
                or record.get('verified') is not True or record['fps'] != spec['requested_fps']):
            raise ValueError(f'Invalid catalogue feature assignment: {key}')
    return spec, index


@lru_cache(maxsize=32)
def _cached(root):
    # Each worker uses an immutable catalogue; startup validation rereads bindings.
    return _load(root)


def resolve(root, row, fps, expected_id=None):
    from .features import cache_key
    data = _cached(str(Path(root).resolve()))
    if data is None:
        return str(root), expected_id
    spec, index = data
    if float(fps) != spec['requested_fps'] or (expected_id is not None and expected_id != spec['cache_id']):
        raise ValueError('Feature catalogue sampling/identity mismatch')
    key = cache_key(row, fps)
    if key not in index['entries']:
        raise ValueError(f'Feature catalogue is missing video: {key}')
    entry = index['entries'][key]
    return entry['root'], entry['cache_id']


def validate(root, manifests, fps, *, replacement_root=None):
    from concurrent.futures import ThreadPoolExecutor
    from .features import cache_key
    if replacement_root:
        raise ValueError('A feature catalogue cannot use a separate adaptive replacement cache')
    spec, index = _load(root)
    if float(fps) != spec['requested_fps']:
        raise ValueError('Feature catalogue sampling rate mismatch')
    for manifest in manifests:
        rows = {}
        for line in Path(manifest).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[(row['video_path'], float(row['duration']))] = row
        with ThreadPoolExecutor(max_workers=8) as pool:
            for key in pool.map(lambda row: cache_key(row, fps), rows.values()):
                if key not in index['entries']:
                    raise ValueError(f'Manifest video is missing from catalogue: {key}')
    return spec
