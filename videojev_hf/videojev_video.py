"""Decode original-resolution RGB frames on an exact fps grid; never resize here."""
import numpy as np
import torch
from .videojev_temporal import time_grid


def sampling_grid(duration, fps=2., max_frames=None):
    """Keep the requested fps unless the explicit full-video frame cap applies."""
    requested=time_grid(duration,fps)
    if max_frames is not None:
        if not isinstance(max_frames,int) or max_frames<1:raise ValueError('max_frames must be a positive integer')
        requested=requested[requested<float(duration)]
        if len(requested)>max_frames:
            effective_fps=max_frames/float(duration)
            return np.arange(max_frames,dtype=np.float64)/effective_fps,effective_fps,'uniform_full_video'
    return requested,float(fps),'fixed_fps'


def read_video(row, fps=2., *, max_frames=None):
    import av
    requested,effective_fps,sampling_mode = sampling_grid(row['duration'],fps,max_frames)
    frames, actual = [], []
    previous, previous_time, origin, index, end_time = None, 0., None, 0, 0.
    duplicate_pts = 0
    with av.open(row['video_path']) as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        stream.thread_count = 2
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError(f"Missing video PTS: {row['video_path']}")
            stamp = float(frame.pts * frame.time_base)
            if origin is None:
                origin = stamp
            now = stamp - origin
            if previous is not None and now < previous_time:
                if previous_time - now <= 1e-4:
                    duplicate_pts += 1
                    continue
                raise ValueError(f"Nonmonotonic PTS: {row['video_path']}")
            while index < len(requested) and requested[index] <= now:
                chosen, chosen_time = (previous, previous_time) if previous is not None and abs(previous_time - requested[index]) <= abs(now - requested[index]) else (frame, now)
                frames.append(chosen.to_ndarray(format='rgb24'))
                actual.append(chosen_time)
                index += 1
            previous, previous_time = frame, now
            frame_duration = float(frame.duration * frame.time_base) if frame.duration else 1 / float(stream.average_rate or 30)
            end_time = now + frame_duration
        if previous is None:
            raise ValueError(f"Empty video: {row['video_path']}")
        # Sample only the media's actual supported time range. Do not fabricate
        # repeated frames to fill an annotation duration longer than the video.
        sampled_duration = min(float(row['duration']), end_time)
        times = requested[requested < sampled_duration]
        while index < len(times):
            frames.append(previous.to_ndarray(format='rgb24'))
            actual.append(previous_time)
            index += 1
    if not len(times) or len(frames) < len(times):
        raise ValueError('Video does not cover any requested sample')
    # NumPy's large contig-to-contig copy is exceptionally slow on this host.
    # ATen stacking preserves every uint8 pixel exactly and avoids that path.
    if max_frames is not None and len(times)!=len(requested):
        # A wrong header must not leave a uniform sample covering a fictitious tail.
        if sampled_duration>=float(row['duration']):raise ValueError('Sampling grid extends beyond unchanged media duration')
        return read_video(dict(row,duration=sampled_duration),fps,max_frames=max_frames)
    video = torch.stack([torch.from_numpy(frame) for frame in frames[:len(times)]]).numpy()
    result = dict(video=video, times=times, fps=effective_fps, input_source='video',
                decoded_times=np.asarray(actual[:len(times)], dtype=np.float64),
                media_duration=end_time, sampled_duration=sampled_duration,
                ignored_duplicate_pts=duplicate_pts)
    if max_frames is not None:
        result.update(requested_fps=float(fps),max_frames=max_frames,sampling_mode=sampling_mode,
                      sampling_duration=float(row['duration']))
    return result
