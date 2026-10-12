"""Distinguish an inaccurate duration header from a missing RGB sample."""
import numpy as np
import pytest

from timejev.rgb_media import fingerprint, sampled_chunks, probe_video_geometry


@pytest.fixture
def short_video(tmp_path):
    import av
    path = tmp_path/'short.mp4'
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('mpeg4', rate=25)
        stream.width, stream.height, stream.pix_fmt = 64, 32, 'yuv420p'
        for i in range(24):
            frame = av.VideoFrame.from_ndarray(np.full((32, 64, 3), i*8, np.uint8), format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return dict(source=dict(kind='video', path=str(path), identity=fingerprint(path)),
                frames=4, effective_fps=4., sampled_duration=1., original_hw=[32, 64])


def test_header_one_frame_longer_keeps_all_supported_samples(short_video):
    # Actual end=.96, header end=1.00; requested times 0,.25,.50,.75 are covered.
    chunks = list(sampled_chunks(short_video, chunk_frames=2))
    assert [len(x) for x in chunks] == [2, 2]
    values = [float(frame.float().mean()) for chunk in chunks for frame in chunk]
    assert values == sorted(values) and len(set(values)) == 4


def test_genuinely_missing_final_sample_is_not_padded(short_video):
    # The next requested time is 1.00, beyond the decoded .96-second video.
    record = dict(short_video, frames=5, sampled_duration=1.04)
    with pytest.raises(ValueError, match='last planned sample.*short.mp4'):
        list(sampled_chunks(record, chunk_frames=2))


def test_probe_uses_actual_decoded_timeline(short_video):
    width, height, end_time = probe_video_geometry(short_video['source']['path'])
    assert (width, height) == (64, 32)
    assert end_time == pytest.approx(.96)
