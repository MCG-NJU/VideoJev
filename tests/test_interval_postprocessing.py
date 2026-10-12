"""Offline checks of the default interval protocol and legacy export semantics."""
import copy
import json

import numpy as np
import pytest

from timejev import temporal


UNIT_BOUNDS = np.asarray([[0., 1.], [1., 2.], [2., 3.], [3., 4.]])


def process(prediction, evidence, bounds=UNIT_BOUNDS, duration=4., **kwargs):
    return temporal.postprocess_interval(prediction, evidence, bounds, duration, **kwargs)


def test_half_second_rounding_uses_ties_to_even():
    # Ordinary, nondegenerate endpoints expose both directions of a half tie.
    assert process([.5, 2.5], [0., 0., 0., 0.]) == [0., 2.]
    assert process([1.5, 3.5], [0., 0., 0., 0.]) == [2., 4.]


@pytest.mark.parametrize("evidence, expected", [
    ([0., 4., 1., 0.], [1., 2.]),
    ([0., 1., 4., 0.], [2., 3.]),
])
def test_zero_length_repair_follows_model_evidence_on_either_side(evidence, expected):
    assert process([1.8, 2.2], evidence) == expected


def test_repair_integrates_evidence_over_time_instead_of_counting_cells():
    bounds = [[0., 1.], [1., 1.125], [1.125, 2.], [2., 3.], [3., 4.]]
    # The left candidate contains the largest logit but only 1/8 second of it.
    # Its integral is 1; the right integral is 2.
    assert process([1.9, 2.1], [0., 8., 0., 2., 0.], bounds) == [2., 3.]


@pytest.mark.parametrize("raw, expected", [
    ([1.55, 2.05], [1., 2.]),
    ([1.95, 2.45], [2., 3.]),
    ([1.75, 2.25], [1., 2.]),
])
def test_evidence_ties_use_unrounded_midpoint_then_earlier_start(raw, expected):
    # Both candidate integrals are 2. Raw midpoint is respectively <2, >2, =2.
    assert process(raw, [0., 2., 2., 0.]) == expected


@pytest.mark.parametrize("raw, evidence, bounds, duration, expected", [
    ([.1, .4], [-3., -1.], [[0., 1.], [1., 2.]], 2., [0., 1.]),
    ([1.6, 2.], [-3., -1.], [[0., 1.], [1., 2.]], 2., [1., 2.]),
    ([1.75, 2.4], [0., 0., 5.], [[0., 1.], [1., 2.], [2., 2.4]], 2.4, [2., 2.4]),
    ([.12, .25], [-4.], [[0., .3]], .3, [0., .3]),
    # Rounding gives [1,1], whose valid left neighbor is [0,1], then clipped.
    # Clamping first would instead move the repair anchor to the fraction .7.
    ([.6, .7], [2.], [[0., .7]], .7, [0., .7]),
])
def test_boundary_short_video_and_repair_before_clamp(raw, evidence, bounds, duration, expected):
    assert process(raw, evidence, bounds, duration) == expected


def test_nondegenerate_rounded_interval_only_receives_duration_clamp():
    assert process([.2, 2.8], [-100., 100., -100.],
                   [[0., 1.], [1., 2.], [2., 2.8]], 2.8) == [0., 2.8]


def test_legacy_none_keeps_the_exact_continuous_interval():
    raw = [.375, .875]
    assert process(raw, [-2., -1., -2., -3.], method="none") == raw


def test_decode_default_processes_but_legacy_preserves_native_map():
    times, bounds = temporal.pair_timeline(temporal.time_grid(1.4, 4.), 1.4)
    logits = np.asarray([-3., -1., -2.])
    np.testing.assert_array_equal(
        temporal.decode(logits, times, 1.4, bounds=bounds, postprocess="none"), [.375, .875])
    np.testing.assert_array_equal(
        temporal.decode(logits, times, 1.4, bounds=bounds), [0., 1.])
    assert temporal.POSTPROCESS_VERSION == "round_repair_clamp_v1"


def test_decode_passes_original_smoothed_threshold_shifted_evidence(monkeypatch):
    times, bounds = temporal.pair_timeline(temporal.time_grid(1.4, 4.), 1.4)
    seen = []
    real_postprocess = temporal.postprocess_interval

    def capture(prediction, shifted_logits, cell_bounds, duration, method=temporal.POSTPROCESS_VERSION):
        seen.append((list(prediction), np.asarray(shifted_logits).copy()))
        return real_postprocess(prediction, shifted_logits, cell_bounds, duration, method=method)

    monkeypatch.setattr(temporal, "postprocess_interval", capture)
    actual = temporal.decode([-5., 1., -2.], times, 1.4, threshold=.25, smooth=3, bounds=bounds)
    assert actual == [1., 1.4]
    assert len(seen) == 1
    assert seen[0][0] == [.875, 1.4]
    # Edge-replicated width-3 average is [-3,-2,-1]; threshold .25 adds log(3).
    np.testing.assert_allclose(seen[0][1], np.asarray([-3., -2., -1.]) + np.log(3.), rtol=0, atol=1e-15)


def test_postprocessing_does_not_mutate_saved_prediction_or_evidence():
    raw = np.asarray([1.8, 2.2])
    evidence = np.asarray([0., 4., 1., 0.])
    bounds = UNIT_BOUNDS.copy()
    before = [x.copy() for x in (raw, evidence, bounds)]
    process(raw, evidence, bounds)
    for current, expected in zip((raw, evidence, bounds), before):
        np.testing.assert_array_equal(current, expected)


def test_unknown_protocol_cannot_silently_produce_another_protocol():
    with pytest.raises(ValueError):
        process([.2, 1.2], [0., 0., 0., 0.], method="round_typo")


@pytest.fixture
def three_benchmark_examples():
    names = ("charades", "activitynet", "qvhighlights")
    times, bounds = temporal.pair_timeline(temporal.time_grid(1.4, 4.), 1.4)
    rows = {name: [dict(id=f"{name}/0", video_path=f"/unused/{name}.mp4",
                       query="Find the action.", spans=[[.6, 1.4]], duration=1.4)]
            for name in names}
    scores = {name: [dict(id=f"{name}/0", logits=[-5., -.5, 1.], times=times.tolist(),
                         bounds=bounds.tolist(), sampled_duration=1.4,
                         audit={"score_bounds": bounds.tolist()})]
              for name in names}
    timing = {name: {"seconds": 1.} for name in names}
    return rows, scores, timing


def test_default_selects_continuously_then_postprocesses(three_benchmark_examples):
    from timejev.benchmark import select_decoder

    rows, scores, _ = three_benchmark_examples
    decoder = select_decoder(rows, scores, [.25, .5], [1])
    # Postprocessing reverses the ranking in this example. The default must
    # still choose .25 by continuous IoU, then report its processed output IoU.
    assert decoder['threshold'] == .25
    assert decoder['selection_postprocess'] == 'none'
    assert decoder['postprocess'] == temporal.POSTPROCESS_VERSION
    assert decoder['selection_mean_mIoU'] == pytest.approx(100 * .8 / 1.025)
    assert decoder['mean_mIoU'] == pytest.approx(100 * .4 / 1.4)


@pytest.mark.parametrize("selection_protocol, expected_threshold, candidate_ious", [
    ("none", .25, (.8 / 1.025, .525 / .8)),
    (temporal.POSTPROCESS_VERSION, .5, (.4 / 1.4, .4 / .8)),
])
def test_explicit_selection_policies_keep_selection_and_final_metrics_distinct(
        tmp_path, three_benchmark_examples, selection_protocol, expected_threshold, candidate_ious):
    from timejev.benchmark import export, select_decoder

    rows, scores, timing = three_benchmark_examples
    decoder = select_decoder(rows, scores, [.25, .5], [1],
                             postprocess=temporal.POSTPROCESS_VERSION,
                             selection_postprocess=selection_protocol)
    # The same logits/GT rank .25 first continuously but .5 after processing.
    assert decoder["threshold"] == expected_threshold
    assert decoder["smooth"] == 1
    assert decoder["postprocess"] == temporal.POSTPROCESS_VERSION
    assert decoder["selection_postprocess"] == selection_protocol
    assert decoder["clamp_duration"] == "sampled_duration"
    assert decoder["selection_data"] == list(rows)
    expected_final_iou = .4 / 1.4 if expected_threshold == .25 else .4 / .8
    selected_index = 0 if expected_threshold == .25 else 1
    assert decoder["mean_mIoU"] == pytest.approx(100 * expected_final_iou)
    assert decoder["selection_mean_mIoU"] == pytest.approx(100 * candidate_ious[selected_index])
    for candidate, threshold, expected_iou in zip(decoder["candidates"], [.25, .5], candidate_ious):
        assert candidate["threshold"] == threshold
        assert candidate["selection_postprocess"] == selection_protocol
        assert candidate["mean_mIoU"] == pytest.approx(100 * expected_iou)
        for name in rows:
            assert candidate["metrics"][name]["mIoU"] == pytest.approx(100 * expected_iou)
    for name in rows:
        assert decoder["metrics"][name]["mIoU"] == pytest.approx(100 * expected_final_iou)
        assert decoder["selection_metrics"][name]["mIoU"] == pytest.approx(
            100 * candidate_ious[selected_index])

    summary = export(rows, scores, decoder, tmp_path, timing)
    saved = json.loads((tmp_path / "decoder.json").read_text())
    assert saved == decoder
    for name in rows:
        assert summary[name]["mIoU"] == decoder["metrics"][name]["mIoU"]


def test_new_export_summary_official_and_jsonl_use_the_same_processed_intervals(tmp_path, three_benchmark_examples):
    from timejev.benchmark import export

    rows, scores, timing = three_benchmark_examples
    originals = copy.deepcopy((rows, scores))
    decoder = {"threshold": .25, "smooth": 1, "postprocess": temporal.POSTPROCESS_VERSION}
    summary = export(rows, scores, decoder, tmp_path, timing)
    assert json.loads((tmp_path / "benchmark_summary.json").read_text()) == summary
    saved_decoder = json.loads((tmp_path / "decoder.json").read_text())
    assert saved_decoder["postprocess"] == temporal.POSTPROCESS_VERSION
    assert saved_decoder["clamp_duration"] == "sampled_duration"
    assert saved_decoder["mean_mIoU"] == pytest.approx(100 * .4 / 1.4)
    for name in rows:
        persisted = json.loads((tmp_path / f"{name}_metrics.json").read_text())
        assert persisted == summary[name]
        assert persisted["n"] == 1 and persisted["coverage"] == "1/1"
        assert persisted["mIoU"] == pytest.approx(100 * .4 / 1.4)
        assert [persisted[key] for key in ("R1@0.3", "R1@0.5", "R1@0.7")] == [0., 0., 0.]
        assert persisted["postprocess"] == temporal.POSTPROCESS_VERSION
        assert saved_decoder["metrics"][name] == {
            key: persisted[key] for key in ("n", "R1@0.3", "R1@0.5", "R1@0.7", "mIoU")}
        official = json.loads((tmp_path / f"{name}_official.json").read_text())
        assert len(official) == 1
        entry = next(iter(official.values()))
        assert entry["timestamps"] == [[0., 1.]]
        assert entry["postprocess"] == temporal.POSTPROCESS_VERSION
        record = json.loads((tmp_path / f"{name}_predictions.jsonl").read_text())
        assert record["prediction"] == entry["timestamps"][0]
        assert record["raw_prediction"] == [.375, 1.4]
        assert record["postprocess"] == temporal.POSTPROCESS_VERSION
    assert (rows, scores) == originals


@pytest.mark.parametrize("explicit_none", [False, True])
def test_old_decoder_exports_keep_legacy_intervals(tmp_path, three_benchmark_examples, explicit_none):
    from timejev.benchmark import export

    rows, scores, timing = three_benchmark_examples
    decoder = {"threshold": .5, "smooth": 1}
    if explicit_none:
        decoder["postprocess"] = "none"
    summary = export(rows, scores, decoder, tmp_path, timing)
    for name in rows:
        official = json.loads((tmp_path / f"{name}_official.json").read_text())
        entry = next(iter(official.values()))
        record = json.loads((tmp_path / f"{name}_predictions.jsonl").read_text())
        assert entry["timestamps"] == [[.875, 1.4]]
        assert record["prediction"] == [.875, 1.4]
        assert record["raw_prediction"] == record["prediction"]
        assert summary[name]["mIoU"] == pytest.approx(100 * .525 / .8)
        assert [summary[name][key] for key in ("R1@0.3", "R1@0.5", "R1@0.7")] == [100., 100., 0.]
        assert summary[name]["postprocess"] == entry["postprocess"] == record["postprocess"] == "none"
