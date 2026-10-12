"""Bounded RGB decoding from timestamped JPEG directories or original videos.

Only sampling/shape metadata may be persisted. No learned video features are read.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

FRAME_ROOTS = {name: str(Path(__file__).resolve().parents[1]/'data/frames'/name)
               for name in ('lvbench', 'videomme', 'longvideobench', 'mlvu')}


def route(row, benchmark=None):
    """Use an explicit frame source, conventional benchmark frames, or raw video."""
    if 'media_source' in row:
        source = dict(row['media_source'])
        if source.get('kind') not in ('video', 'frames_sidecar', 'frames_4fps'):
            raise ValueError('Unknown media_source kind')
        return source
    video = Path(row['video_path'])
    if benchmark in FRAME_ROOTS:
        stem = Path(video.name.removesuffix('.pre_audio_pad')).stem
        frames = Path(FRAME_ROOTS[benchmark])/stem
        if frames.is_dir():
            return dict(kind='frames_sidecar', path=str(frames))
    return dict(kind='video', path=str(video))


def fingerprint(path):
    stat = Path(path).stat()
    return dict(path=str(Path(path).resolve()), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def probe_video_geometry(path):
    """Probe the decoded tail, since MP4 duration headers can include a phantom frame."""
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type, stream.thread_count = 'AUTO', 2
        first = next(container.decode(stream))
        if first.pts is None:
            raise ValueError(f'Missing initial video PTS: {path}')
        width, height = first.width, first.height
        origin = float(first.pts*first.time_base)
        header = (float(stream.duration*stream.time_base) if stream.duration else
                  float(container.duration/av.time_base))
        container.seek(max(0, int((header-2+origin)/float(stream.time_base))),
                       stream=stream, backward=True)
        last = None
        for frame in container.decode(stream):
            last = frame
        if last is None:
            # A bad header may seek beyond the final keyframe; decode from start.
            container.seek(int(origin/float(stream.time_base)), stream=stream, backward=True)
            for frame in container.decode(stream):
                last = frame
        if last is None or last.pts is None:
            raise ValueError(f'Cannot establish decoded video end: {path}')
        frame_duration = (float(last.duration*last.time_base) if last.duration else
                          1/float(stream.average_rate or 30))
        end_time = float(last.pts*last.time_base)-origin+frame_duration
        if end_time <= 0:
            raise ValueError(f'Invalid decoded video duration: {path}')
    return width, height, end_time


def metadata(row, benchmark=None):
    """Read the extraction completion record, or probe geometry without full decode."""
    source = route(row, benchmark)
    path = Path(source['path'])
    if not path.exists():
        raise FileNotFoundError(f'RGB source missing: {path}')
    fps = 4. if benchmark and benchmark not in FRAME_ROOTS else 2.
    if source['kind'] == 'frames_sidecar':
        complete = path/'.complete.json'
        if not complete.is_file() or not (path/'.timestamps.jsonl').is_file():
            raise FileNotFoundError(f'RGB extraction not complete: {path}')
        state = json.loads(complete.read_text())
        conf = state['config']
        if (conf['fps'] != 2 or conf['max_frames'] != 2048 or
                conf['cap_mode'] != 'uniform_full_video' or conf['resize']):
            raise ValueError(f'Unexpected RGB extraction protocol: {path}')
        if Path(state['source']['path']).resolve() != Path(row['video_path']).resolve():
            # MLVU preserves one pre-audio-remux file in its official manifest.
            # Accept its extracted frames only after verifying identical video
            # packet payloads; matching stems or durations alone are insufficient.
            import subprocess
            old, new = row['video_path'], state['source']['path']
            if not old.endswith('.pre_audio_pad') or str(Path(old.removesuffix('.pre_audio_pad')).resolve()) != str(Path(new).resolve()):
                raise ValueError(f'Frame basename maps to another source video: {path}')
            hashes = [subprocess.check_output(['ffmpeg', '-v', 'error', '-i', name, '-map', '0:v:0',
                      '-c', 'copy', '-f', 'hash', '-hash', 'sha256', '-'], text=True).strip() for name in (old, new)]
            if hashes[0] != hashes[1]:
                raise ValueError(f'Pre/post audio remux video payload differs: {path}')
            source['verified_video_stream_alias'] = dict(original=old, extracted_from=new, sha256=hashes[0])
        if len(state['dimensions']) != 1:
            raise ValueError(f'Variable RGB dimensions require an explicit policy: {path}')
        width, height = state['dimensions'][0]
        duration = min(float(row['duration']), float(state['media_duration']))
        effective = float(state['effective_fps'])
        frames = min(int(state['frame_count']), int(math.ceil(duration*effective-1e-9)))
        # The sidecar is authoritative, including a corrected media duration.
        source['identity'] = fingerprint(complete)
        source['timestamps_identity'] = fingerprint(path/'.timestamps.jsonl')
        mode = state['sampling_mode']
    else:
        if source['kind'] == 'frames_4fps':
            files = sorted(path.glob('*.jpg'))
            if not files:
                raise FileNotFoundError(f'No RGB frames: {path}')
            with Image.open(files[0]) as image:
                width, height = image.size
            last = float(files[-1].stem.replace('_', '.'))
            duration = min(float(row['duration']), last+.25)
            source.update(first=files[0].name, last=files[-1].name, frame_count=len(files),
                          identity=fingerprint(files[0]))
        else:
            width, height, media_duration = probe_video_geometry(path)
            duration = min(float(row['duration']), media_duration)
            source['identity'] = fingerprint(path)
        from .video import sampling_grid
        times, effective, mode = sampling_grid(duration, fps, 2048)
        frames = len(times)
    if frames < 1 or frames > 2048 or duration <= 0:
        raise ValueError(f'Invalid RGB frame count/duration: {path}')
    return dict(video_key=row['video_key'], source=source, frames=frames, fps=fps,
                effective_fps=effective, original_hw=[height, width],
                sampled_duration=duration, media_duration=duration, sampling_mode=mode)


def timeline(record):
    from .temporal import pair_timeline
    times = np.arange(record['frames'], dtype=np.float64)/record['effective_fps']
    centers, bounds = pair_timeline(times, record['sampled_duration'])
    return dict(record, score_times=centers.tolist(), score_bounds=bounds.tolist())


def _frame_paths(record):
    source, n = record['source'], record['frames']
    root = Path(source['path'])
    times = np.arange(n, dtype=np.float64)/record['effective_fps']
    if source['kind'] == 'frames_sidecar':
        for name, key in (('.complete.json', 'identity'), ('.timestamps.jsonl', 'timestamps_identity')):
            if fingerprint(root/name) != source[key]:
                raise ValueError(f'RGB extraction metadata changed: {root/name}')
        stamps = [json.loads(line) for line in (root/'.timestamps.jsonl').read_text().splitlines()]
        if len(stamps) < n or not np.allclose([x['requested_seconds'] for x in stamps[:n]], times,
                                             rtol=1e-10, atol=1e-8):
            raise ValueError(f'RGB timestamps differ from planned sampling: {root}')
        return [root/x['filename'] for x in stamps[:n]]
    # Select nearest extracted 4fps frame; ties use the earlier frame. Requested
    # timestamps remain the exact 2fps/full-video grid, never rounded filenames.
    files = sorted(root.glob('*.jpg'))
    if (len(files) != source['frame_count'] or files[-1].name != source['last'] or
            fingerprint(files[0]) != source['identity']):
        raise ValueError(f'Grounding RGB directory changed: {root}')
    available = np.array([float(p.stem.replace('_', '.')) for p in files])
    positions = np.searchsorted(available, times).clip(0, len(files)-1)
    left = (positions-1).clip(0)
    positions = np.where(abs(available[left]-times) <= abs(available[positions]-times), left, positions)
    return [files[i] for i in positions]


def sampled_chunks(record, chunk_frames=32):
    """Decode at most one small frame chunk; original videos are streamed once."""
    if chunk_frames < 2 or chunk_frames % 2:
        raise ValueError('Chunk boundaries must preserve native temporal pairs')
    pending = []
    def tensor(frames):
        return torch.stack([torch.from_numpy(x) for x in frames]).permute(0, 3, 1, 2)
    if record['source']['kind'] != 'video':
        for path in _frame_paths(record):
            with Image.open(path) as image:
                image = image.convert('RGB')
                if [image.height, image.width] != record['original_hw']:
                    raise ValueError(f'RGB shape changed: {path}')
                pending.append(np.array(image))
            if len(pending) == chunk_frames:
                yield tensor(pending)
                pending = []
    else:
        import av
        source = record['source']
        if fingerprint(source['path']) != source['identity']:
            raise ValueError('Original evaluation video changed')
        times = np.arange(record['frames'])/record['effective_fps']
        index, previous, previous_time, origin, end_time = 0, None, 0., None, 0.
        with av.open(source['path']) as container:
            stream = container.streams.video[0]
            stream.thread_type, stream.thread_count = 'AUTO', 2
            for frame in container.decode(stream):
                if frame.pts is None:
                    raise ValueError('Missing source video PTS')
                stamp = float(frame.pts*frame.time_base)
                origin = stamp if origin is None else origin
                now = stamp-origin
                if previous is not None and now < previous_time:
                    if previous_time-now <= 1e-4:
                        continue
                    raise ValueError('Nonmonotonic source video PTS')
                while index < len(times) and times[index] <= now:
                    chosen = (previous if previous is not None and
                              abs(previous_time-times[index]) <= abs(now-times[index]) else frame)
                    pending.append(chosen.to_ndarray(format='rgb24'))
                    index += 1
                    if len(pending) == chunk_frames:
                        yield tensor(pending)
                        pending = []
                previous, previous_time = frame, now
                end_time = now+(float(frame.duration*frame.time_base) if frame.duration else
                                1/float(stream.average_rate or 30))
            if previous is None:
                raise ValueError(f"Empty original evaluation video: {source['path']}")
            # MP4 headers can extend a frame beyond the decoded stream. Every
            # requested sample must be supported by real media; a header-only
            # tail after the final requested sample does not remove any input.
            if times[-1] >= end_time:
                raise ValueError(f"Video ends before the last planned sample: {source['path']}; "
                                 f"last_requested={times[-1]:.9f}, decoded_end={end_time:.9f}")
            while index < len(times):
                pending.append(previous.to_ndarray(format='rgb24'))
                index += 1
                if len(pending) == chunk_frames:
                    yield tensor(pending)
                    pending = []
    if pending:
        yield tensor(pending)


def preprocess(processor, record, chunk_frames=32):
    """Native resize/normalize/patchify in CPU workers, on complete frame pairs."""
    from transformers.image_utils import SizeDict
    vp, chunks, groups = processor.video_processor, [], 0
    size = SizeDict(height=record['processed_hw'][0], width=record['processed_hw'][1])
    for frames in sampled_chunks(record, chunk_frames):
        resized = vp.resize(frames, size=size, resample=vp.resample)
        native = vp(videos=[resized], do_resize=False, do_sample_frames=False, return_tensors='pt')
        grid = native['video_grid_thw'][0].tolist()
        if grid[1:] != record['grid_thw'][1:]:
            raise ValueError('Native RGB processor spatial grid differs from plan')
        groups += grid[0]
        # BF16 matches the native vision autocast input. No features are saved.
        chunks.append(native['pixel_values_videos'].to(torch.bfloat16))
    if groups != record['grid_thw'][0]:
        raise ValueError('Native RGB temporal groups differ from plan')
    return chunks
