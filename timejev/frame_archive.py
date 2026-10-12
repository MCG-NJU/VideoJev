"""One ZIP per video: lossless Zstandard RGB frames, decoded wholly in memory.

No resize, JPEG encoding, learned feature cache, or temporal subsampling occurs.
ZIP_STORED avoids double compression. Zstandard verifies each frame's checksum.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import warnings
import zipfile

import numpy as np
import torch
import zstandard as zstd

from .temporal import time_grid

ARCHIVE_PROTOCOL = 'original_rgb_zstd_frames_zip_v1'


def archive_path(root, row, fps):
    identity = json.dumps([row['video_key'], str(Path(row['video_path']).resolve()), float(fps)])
    key = hashlib.sha256(identity.encode()).hexdigest()[:32]
    return Path(root) / row['video_key'].split('/')[1] / (key + '.zip')


def fingerprint(path):
    stat = Path(path).stat()
    return dict(path=str(Path(path).resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def validate_index(root, manifests, fps):
    index = json.loads((Path(root) / 'index.json').read_text())
    if not index.get('complete') or index['protocol'] != ARCHIVE_PROTOCOL or index['fps'] != fps:
        raise ValueError('Finish the matching frame archive build before evaluation/training')
    for manifest in manifests:
        recorded = index['manifests'].get(str(manifest))
        if recorded != hashlib.sha256(Path(manifest).read_bytes()).hexdigest():
            raise ValueError(f'Archive index/benchmark manifest mismatch: {manifest}')
    return index


def frame_names(count):
    return [f'frames/{i:06d}.rgb.zst' for i in range(count)]


def decode_frame(payload, frame_bytes):
    if zstd.frame_content_size(payload) != frame_bytes:
        raise ValueError('Compressed RGB frame size mismatch')
    return zstd.ZstdDecompressor().decompress(payload, max_output_size=frame_bytes, allow_extra_data=False)


def assemble(parts, shape):
    # Streaming readinto hits this host's pathological large memcpy path. Direct
    # decompression + ATen stacking is byte-identical and avoids that extra copy.
    # frombuffer only reads immutable bytes here; stack creates writable storage.
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='The given buffer is not writable')
        frames = [torch.frombuffer(part, dtype=torch.uint8) for part in parts]
        return torch.stack(frames).numpy().reshape(shape)


def _metadata_check(metadata, row, fps, check_source):
    if metadata['protocol'] != ARCHIVE_PROTOCOL or metadata['fps'] != fps:
        raise ValueError('Archive protocol/fps mismatch')
    if metadata['video_key'] != row['video_key']:
        raise ValueError('Archive video identity mismatch')
    if metadata['requested_duration'] < row['duration']:
        raise ValueError('Archive does not cover the requested annotation duration')
    if check_source and metadata['source'] != fingerprint(row['video_path']):
        raise ValueError('Source video changed; rebuild its frame archive')


def _sample(metadata, video, row, fps):
    duration = min(float(row['duration']), metadata['media_duration'])
    times = time_grid(duration, fps)
    if not np.array_equal(times, np.asarray(metadata['times'])[:len(times)]):
        raise ValueError('Archive timestamps do not cover the requested fps grid')
    return dict(video=video[:len(times)], times=times, fps=float(fps),
                decoded_times=np.asarray(metadata['decoded_times'], dtype=np.float64)[:len(times)],
                media_duration=metadata['media_duration'], sampled_duration=duration,
                ignored_duplicate_pts=metadata['ignored_duplicate_pts'], input_source='frame_archive')


def read_archive(row, root, fps=4., *, check_source=True, verify_rgb=False):
    path = archive_path(root, row, fps)
    # A single sequential file read; every subsequent operation is in RAM.
    blob = path.read_bytes()
    data = memoryview(blob)
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        metadata = json.loads(archive.read('metadata.json'))
        _metadata_check(metadata, row, fps, check_source)
        shape = metadata['shape']
        frame_bytes = int(np.prod(shape[1:]))
        parts = []
        names = frame_names(shape[0])
        if len(archive.namelist()) != len(names) + 1 or set(archive.namelist()) != {'metadata.json', *names}:
            raise ValueError('Unexpected or missing archive members')
        for index, name in enumerate(names):
            member = archive.getinfo(name)
            if member.compress_type != zipfile.ZIP_STORED or member.flag_bits & 1:
                raise ValueError('Expected unencrypted ZIP_STORED RGB members')
            name_len, extra_len = struct.unpack_from('<HH', blob, member.header_offset + 26)
            offset = member.header_offset + 30 + name_len + extra_len
            parts.append(decode_frame(data[offset:offset + member.compress_size], frame_bytes))
        video = assemble(parts, shape)
    del parts
    if verify_rgb and hashlib.sha256(memoryview(video)).hexdigest() != metadata['rgb_sha256']:
        raise ValueError('Decoded RGB checksum mismatch')
    return _sample(metadata, video, row, fps)


def write_archive(row, sample, root, *, level=1):
    fps = sample['fps']
    path = archive_path(root, row, fps)
    path.parent.mkdir(parents=True, exist_ok=True)
    before = fingerprint(row['video_path'])
    video = sample['video']
    metadata = dict(protocol=ARCHIVE_PROTOCOL, fps=fps, video_key=row['video_key'], source=before,
                    requested_duration=float(row['duration']), shape=list(video.shape), dtype='uint8',
                    times=sample['times'].tolist(), decoded_times=sample['decoded_times'].tolist(),
                    media_duration=sample['media_duration'], sampled_duration=sample['sampled_duration'],
                    ignored_duplicate_pts=sample['ignored_duplicate_pts'], compression_level=level,
                    rgb_sha256=hashlib.sha256(memoryview(video)).hexdigest())
    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
    try:
        compressor = zstd.ZstdCompressor(level=level, write_checksum=True)
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            archive.writestr('metadata.json', json.dumps(metadata))
            for name, frame in zip(frame_names(len(video)), video):
                archive.writestr(name, compressor.compress(memoryview(frame)))
        if fingerprint(row['video_path']) != before:
            raise ValueError('Source changed during archive construction')
        os.replace(temporary, path)
        # Validate every newly built archive, not just a small sample of the cache.
        restored = read_archive(row, root, fps, verify_rgb=True)
        if not np.array_equal(restored['decoded_times'], sample['decoded_times']):
            raise ValueError('PTS roundtrip mismatch')
    finally:
        temporary.unlink(missing_ok=True)
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, raw_bytes=video.nbytes, fps=fps,
                protocol=ARCHIVE_PROTOCOL,
                frames=len(video), rgb_sha256=metadata['rgb_sha256'], source=before,
                requested_duration=row['duration'], verified=True)


def unpack_loose(row, root, destination, fps=4.):
    """Speed-control fixture: identical compressed frame bytes as the ZIP path."""
    directory = Path(destination) / archive_path(root, row, fps).stem
    directory.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path(root, row, fps)) as archive:
        for name in archive.namelist():
            target = directory / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))
    return directory


def read_loose(row, directory, fps=4.):
    directory = Path(directory)
    metadata = json.loads((directory / 'metadata.json').read_text())
    _metadata_check(metadata, row, fps, True)
    frame_bytes = int(np.prod(metadata['shape'][1:]))
    parts = [decode_frame((directory / name).read_bytes(), frame_bytes)
             for name in frame_names(metadata['shape'][0])]
    video = assemble(parts, metadata['shape'])
    sample = _sample(metadata, video, row, fps)
    sample['input_source'] = 'loose_zstd_frames'
    return sample
