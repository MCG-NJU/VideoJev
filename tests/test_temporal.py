import itertools

import numpy as np

from timejev.temporal import decode, frame_labels, interval_iou, metrics, time_grid, pair_timeline


def test_grid_and_inclusive_labels_at_both_endpoints():
    np.testing.assert_array_equal(time_grid(1.2), [0, .5, 1])
    np.testing.assert_array_equal(frame_labels(time_grid(2), [[.5, 1], [1.5, 2]]), [0, 1, 1, 1])
    np.testing.assert_array_equal(frame_labels([.125, .625, 1.125], [[.125, .625]]), [1, 1, 0])


def test_pair_midpoints_even_odd_and_fps():
    train, bounds = pair_timeline(time_grid(2, 2), 2)
    np.testing.assert_array_equal(train, [.25, 1.25])
    np.testing.assert_array_equal(bounds, [[0, .75], [.75, 2]])
    evaluation, _ = pair_timeline(time_grid(2, 4), 2)
    np.testing.assert_array_equal(evaluation, [.125, .625, 1.125, 1.625])
    odd, bounds = pair_timeline([0, .25, .5], .6)
    np.testing.assert_array_equal(odd, [.125, .5])
    np.testing.assert_array_equal(bounds, [[0, .375], [.375, .6]])


def test_decoder_matches_exhaustive_map():
    rng = np.random.default_rng(12)
    for length in range(1, 20):
        logits = rng.normal(size=length)
        times, bounds = pair_timeline(time_grid(length / 2, 4), length / 2)
        for threshold in [.15, .25, .5, .65]:
            scores = logits - np.log(threshold / (1 - threshold))
            candidates = [(scores[a:b].sum(), a, b) for a in range(length) for b in range(a + 1, length + 1)]
            _, a, b = max(candidates)
            assert decode(logits, times, length / 2, threshold, bounds=bounds, postprocess="none") == [bounds[a, 0], bounds[b-1, 1]]


def test_negative_and_terminal_intervals():
    times, bounds = pair_timeline(time_grid(1.4, 4), 1.4)
    assert decode([-3, -1, -2], times, 1.4, bounds=bounds, postprocess="none") == [.375, .875]
    assert decode([-3, -1, 2], times, 1.4, threshold=.5, bounds=bounds, postprocess="none") == [.875, 1.4]
    assert interval_iou([0, 2], [1, 3]) == 1 / 3
    assert interval_iou([0, 2], [2, 4]) == 0


def test_metric_known_case():
    result = metrics([[0, 2], [5, 6]], [{'spans': [[0, 2]]}, {'spans': [[0, 1]]}])
    assert result['mIoU'] == 50
    assert result['R1@0.7'] == 50


def test_grounding_default_and_explicit_threshold_override():
    from videojev_hf.videojev_temporal import decode as hf_decode
    logits = [-2., -.8, .8, -.8, -2.]
    times = [.5, 1.5, 2.5, 3.5, 4.5]
    bounds = [[float(i), float(i+1)] for i in range(5)]
    for decoder in (decode, hf_decode):
        assert decoder(logits, times, 5., bounds=bounds) == [1., 4.]
        assert decoder(logits, times, 5., threshold=.5, bounds=bounds) == [2., 3.]
