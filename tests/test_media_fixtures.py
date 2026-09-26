from __future__ import annotations

from fractions import Fraction

import pytest

from tests.fixtures import FIXTURES, media_params
from tests.fixtures.media import audio_streams, keyframe_times, read_frames, runs, video_stream

pytestmark = pytest.mark.media

CODEC_NAMES = {"h264": "h264", "h265": "hevc", "prores": "prores"}


def test_fixture_matrix_covers_required_variety():
    assert {s.codec for s in FIXTURES} >= {"h264", "h265", "prores"}
    assert len({s.fps for s in FIXTURES}) >= 5
    assert len({(s.width, s.height) for s in FIXTURES}) >= 5
    assert {s.keyframes for s in FIXTURES} == {"cuts", "gop", "sparse", "intra"}
    assert any(not s.audio for s in FIXTURES)
    assert any(len(s.audio) > 1 for s in FIXTURES)
    assert len({s.total_frames for s in FIXTURES}) >= 5


@pytest.mark.parametrize("spec", media_params())
def test_stream_properties(media_files, spec):
    path = media_files[spec.name]
    v = video_stream(path)
    assert v["codec_name"] == CODEC_NAMES[spec.codec]
    assert (v["width"], v["height"]) == (spec.width, spec.height)
    assert Fraction(v["avg_frame_rate"]) == spec.rate
    assert int(v["nb_frames"]) == spec.total_frames

    audio = audio_streams(path)
    assert len(audio) == len(spec.audio)
    for stream, track in zip(audio, spec.audio):
        assert stream["codec_name"] == track.codec
        assert int(stream["channels"]) == track.channels
        assert int(stream["sample_rate"]) == track.sample_rate
        if track.language:
            assert stream["tags"]["language"] == track.language


@pytest.mark.parametrize("spec", media_params())
def test_keyframe_layout(media_files, spec):
    assert keyframe_times(media_files[spec.name]) == pytest.approx(spec.keyframe_secs, abs=1e-4)


@pytest.mark.parametrize("spec", media_params())
def test_frames_alternate_blue_red_with_readable_frame_numbers(media_files, spec):
    frames = read_frames(media_files[spec.name])
    assert [f.number for f in frames] == list(range(spec.total_frames))
    assert [f.color for f in frames] == [spec.frame_color(f.number) for f in frames]
    assert runs([f.color for f in frames]) == [
        (spec.segment_color(i), n) for i, n in enumerate(spec.segments)
    ]
