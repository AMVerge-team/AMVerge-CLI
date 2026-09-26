from __future__ import annotations

import pytest

from amverge.core.detection.keyframe import detect_cuts_by_keyframe, detect_scenes_by_keyframe
from amverge.core.keyframes import generate_keyframes
from amverge.core.keyframes.keyframe_align import get_keyframe_timestamps_pyav
from tests.fixtures import BY_NAME, media_params
from tests.fixtures.media import keyframe_times

pytestmark = pytest.mark.media


@pytest.mark.parametrize("spec", media_params())
def test_pyav_keyframes_match_ffprobe(media_files, spec):
    path = str(media_files[spec.name])
    got = get_keyframe_timestamps_pyav(path)
    assert got == pytest.approx(keyframe_times(path), abs=1e-6)
    assert got == pytest.approx(spec.keyframe_secs, abs=spec.pts_tolerance)


@pytest.mark.parametrize("spec", media_params())
def test_generate_keyframes_matches_layout(media_files, spec):
    got = generate_keyframes(str(media_files[spec.name]))
    assert got == pytest.approx(spec.keyframe_secs, abs=spec.pts_tolerance)


@pytest.mark.parametrize("spec", media_params())
@pytest.mark.parametrize("min_duration", [0.0, 0.25, 1.0])
def test_keyframe_cuts_respect_min_duration(media_files, spec, min_duration):
    cuts = detect_cuts_by_keyframe(str(media_files[spec.name]), min_duration=min_duration)
    kf = spec.keyframe_secs
    assert cuts == sorted(cuts)
    for c in cuts:
        assert min(abs(c - k) for k in kf) < spec.pts_tolerance
    bounds = [0.0, *cuts]
    for a, b in zip(bounds, bounds[1:]):
        assert b - a >= min_duration - 1e-6


def test_keyframe_cuts_on_aligned_source_equal_scene_cuts(media_files):
    spec = BY_NAME["h264_24_cuts"]
    cuts = detect_cuts_by_keyframe(str(media_files[spec.name]), min_duration=0.25)
    assert cuts == pytest.approx(spec.cut_secs, abs=1e-4)


def test_keyframe_min_duration_drops_three_frame_scene(media_files):
    spec = BY_NAME["h264_30_short"]
    path = str(media_files[spec.name])
    assert detect_cuts_by_keyframe(path, min_duration=0.0) == pytest.approx(spec.cut_secs, abs=1e-4)
    assert detect_cuts_by_keyframe(path, min_duration=0.25) == pytest.approx([2.0, 4.0], abs=1e-4)


def test_sparse_source_has_no_keyframe_cuts(media_files):
    spec = BY_NAME["h264_25_sparse"]
    assert detect_cuts_by_keyframe(str(media_files[spec.name])) == []


@pytest.mark.parametrize("spec", media_params())
def test_keyframe_scenes_tile_the_whole_video(media_files, spec):
    scenes = detect_scenes_by_keyframe(str(media_files[spec.name]), min_duration=0.25)
    assert scenes[0][0] == 0.0
    assert scenes[-1][1] == pytest.approx(spec.duration, abs=0.05)
    for (_, end), (start, _) in zip(scenes, scenes[1:]):
        assert end == start
    for start, end in scenes:
        assert end > start
