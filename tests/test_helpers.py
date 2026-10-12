import numpy as np
from timejev.temporal import pair_timeline

def record_for(frames=4, fps=2., effective_fps=None):
    actual_fps = fps if effective_fps is None else effective_fps
    duration = frames / actual_fps
    times, bounds = pair_timeline(np.arange(frames, dtype=np.float64) / actual_fps, duration)
    record = dict(grid_thw=[(frames+1)//2,4,6], frames=frames, fps=fps,
        tokens=(frames+1)//2*6, score_times=times.tolist(), sampled_duration=duration,
        cache_id='test', path='synthetic_feature', audit=dict(score_bounds=bounds.tolist()))
    if effective_fps is not None:
        record['effective_fps'] = effective_fps
    return record


