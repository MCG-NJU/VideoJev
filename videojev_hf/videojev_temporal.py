"""Time grids, Bernoulli labels, interval decoding, and TimeLens metrics."""
import numpy as np

POSTPROCESS_VERSION = 'round_repair_clamp_v1'


def postprocess_interval(prediction, shifted_logits, bounds, duration,
                         method=POSTPROCESS_VERSION):
    """Round seconds, repair collapsed intervals using evidence, then clamp.

    Evidence must already have the decoder's smoothing and threshold shift.
    No ground truth is used. Python round() uses ties-to-even; the final clamp
    may leave a fractional video endpoint. 'none' preserves legacy intervals.
    """
    if method == 'none':
        return [float(value) for value in prediction]
    if method != POSTPROCESS_VERSION:
        raise ValueError(f'Unknown interval postprocessing: {method}')
    bounds = np.asarray(bounds, dtype=np.float64)
    evidence = np.asarray(shifted_logits, dtype=np.float64)
    rounded = [float(round(float(value))) for value in prediction]
    if rounded[1] <= rounded[0]:
        second = rounded[0]
        candidates = [[second - 1, second], [second, second + 1]]
        candidates = [span for span in candidates
                      if span[0] >= 0 and span[0] < bounds[-1, 1]
                      and span[1] > bounds[0, 0]]
        original_midpoint = sum(prediction) / 2

        def score(span):
            overlap = np.maximum(0, np.minimum(bounds[:, 1], span[1])
                                 - np.maximum(bounds[:, 0], span[0]))
            return (float(np.dot(evidence, overlap)),
                    -abs(sum(span) / 2 - original_midpoint), -span[0])

        if candidates:
            rounded = list(max(candidates, key=score))
    return [min(float(duration), max(0., value)) for value in rounded]


def time_grid(duration, fps=2.):
    if duration <= 0 or fps <= 0:
        raise ValueError('duration and fps must be positive')
    # Fixed-rate grid, never linspace (which silently changes fps).
    return np.arange(int(np.ceil(duration * fps)), dtype=np.float64) / fps


def frame_labels(times, spans):
    times = np.asarray(times)
    labels = np.zeros(len(times), dtype=np.float32)
    for start, end in spans:
        labels[(times >= start) & (times <= end)] = 1
    return labels


def pair_timeline(frame_times, duration):
    """Native temporal patch=2: one score at each pair's midpoint.

    The official odd-frame duplicate gives the last group its real frame time.
    Interval cell boundaries lie halfway between adjacent original frame times;
    the outer cells extend to the observed video's endpoints.
    """
    t = np.asarray(frame_times, dtype=np.float64)
    if t.ndim != 1 or not len(t) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError('Expected a nonempty strictly increasing frame time grid')
    if t[0] < 0 or t[-1] >= duration:
        raise ValueError('Frame timestamps must lie within the sampled video')
    first = t[::2]
    second = t[np.minimum(np.arange(0, len(t), 2) + 1, len(t) - 1)]
    centers = (first + second) / 2
    internal = (t[1:-1:2] + t[2::2]) / 2
    edges = np.concatenate(([0.], internal, [float(duration)]))
    return centers, np.stack((edges[:-1], edges[1:]), axis=1)


def interval_iou(a, b):
    # Identical to TimeLens utils.iou for valid, single intervals.
    if a[1] <= a[0] or b[1] <= b[0]:
        return 0.
    return max(0., min(a[1], b[1]) - max(a[0], b[0])) / (max(a[1], b[1]) - min(a[0], b[0]))


def decode(logits, times, duration, threshold=.25, smooth=1, *, bounds,
           postprocess=POSTPROCESS_VERSION, return_raw=False):
    """Maximum Bernoulli-likelihood nonempty contiguous segment (Kadane, O(T)).

    Relative to predicting background everywhere, assigning a frame foreground
    contributes log(p/(1-p)) - log(threshold/(1-threshold)). A calibrated .5
    threshold is exactly the constrained independent-Bernoulli MAP segment.
    All-negative sequences return their strongest single frame, without using GT.
    Each timestamp is a pair midpoint, with explicit temporal cell bounds.
    The resulting segment receives the default round/repair/clamp protocol;
    postprocess='none' returns the original continuous MAP interval.
    return_raw=True also returns that continuous interval for export provenance.
    """
    z = np.asarray(logits, dtype=np.float64)
    times = np.asarray(times, dtype=np.float64)
    bounds = np.asarray(bounds, dtype=np.float64)
    if len(z) == 0 or len(z) != len(times) or not np.isfinite(z).all():
        raise ValueError('nonempty finite scores must match timestamps')
    if (bounds.shape != (len(z), 2) or not np.isfinite(bounds).all()
            or np.any(bounds[:, 0] > times) or np.any(bounds[:, 1] < times)
            or np.any(bounds[:, 1] <= bounds[:, 0]) or bounds[0, 0] != 0
            or bounds[-1, 1] != duration
            or not np.array_equal(bounds[1:, 0], bounds[:-1, 1])):
        raise ValueError('Invalid pair score interval bounds')
    if not 0 < threshold < 1 or smooth < 1 or smooth % 2 != 1:
        raise ValueError('threshold in (0,1); smoothing width must be positive and odd')
    if smooth > 1:
        z = np.convolve(np.pad(z, (smooth // 2, smooth // 2), mode='edge'),
                        np.ones(smooth) / smooth, mode='valid')
    z = z - np.log(threshold / (1 - threshold))
    best = current = float(z[0])
    start = best_start = best_end = 0
    for i in range(1, len(z)):
        if current <= 0:
            start, current = i, float(z[i])
        else:
            current += float(z[i])
        if current > best:
            best, best_start, best_end = current, start, i
    prediction = [float(bounds[best_start, 0]), float(bounds[best_end, 1])]
    processed = postprocess_interval(prediction, z, bounds, duration, method=postprocess)
    return (processed, prediction) if return_raw else processed


def metrics(predictions, rows):
    if len(predictions) != len(rows):
        raise ValueError('coverage mismatch')
    ious = [max(interval_iou(p, s) for s in r['spans']) for p, r in zip(predictions, rows)]
    return dict(n=len(rows), **{f'R1@{t}': float(100 * np.mean(np.asarray(ious) >= t)) for t in [.3, .5, .7]},
                mIoU=float(100 * np.mean(ious)))
