from __future__ import annotations

import os
from pathlib import Path

import pytest

from amverge.core.cutting.segmenter import collect_scenes, run_ffmpeg_segment
from amverge.core.cutting.smart_cut import cut_all_scenes, cut_scene, snap_range_to_keyframes
from amverge.core.detection.short_scenes import merge_short_scenes
from amverge.core.keyframes.keyframe_align import get_keyframe_timestamps_pyav
from amverge.core.thumbnails import make_thumbnail
from tests.fixtures import BY_NAME, media_params
from tests.fixtures.media import jpeg_color, read_frames, video_stream


class TestSnapRange:
    KF = [0.0, 1.0, 2.0, 3.0]

    def test_aligned_range_is_unchanged(self):
        assert snap_range_to_keyframes(self.KF, 1.0, 2.0) == (1.0, 2.0)

    def test_widens_outward(self):
        assert snap_range_to_keyframes(self.KF, 1.4, 2.6) == (1.0, 3.0)

    def test_end_past_last_keyframe_runs_to_eof(self):
        assert snap_range_to_keyframes(self.KF, 2.5, 9.0) == (2.0, None)

    def test_requires_keyframes(self):
        with pytest.raises(ValueError):
            snap_range_to_keyframes([], 0.0, 1.0)

    def test_float_noise_does_not_cross_a_keyframe(self):
        kf = [0.0, 4.5045, 9.009]
        assert snap_range_to_keyframes(kf, 4.5045 - 1e-12, 9.009 + 1e-12) == (4.5045, 9.009)

    def test_requires_keyframe_before_start(self):
        with pytest.raises(ValueError):
            snap_range_to_keyframes([1.0, 2.0], 0.5, 1.5)


@pytest.mark.media
@pytest.mark.parametrize("spec", media_params())
def test_reencode_cut_is_frame_exact(media_files, spec, tmp_path):
    path = media_files[spec.name]
    for i, (a, b) in enumerate(spec.scene_frames):
        clip, mode, offset = cut_scene(
            path, spec.frame_time(a), spec.frame_time(b), i, tmp_path, [], "reencode"
        )
        assert mode == "reencode" and offset == 0.0
        frames = read_frames(clip)
        assert [f.number for f in frames] == list(range(a, b)), f"scene {i}"
        assert {f.color for f in frames} == {spec.segment_color(i)}


@pytest.mark.media
@pytest.mark.parametrize("spec", media_params("h265_60_main10_gop", "prores_23976_422hq"))
def test_reencode_cut_keeps_10bit(media_files, spec, tmp_path):
    a, b = spec.scene_frames[1]
    clip, _, _ = cut_scene(media_files[spec.name], spec.frame_time(a), spec.frame_time(b), 1, tmp_path, [], "reencode")
    assert video_stream(clip)["pix_fmt"] == "yuv420p10le"


@pytest.mark.media
@pytest.mark.parametrize("spec", media_params())
def test_copy_cut_snaps_outward_to_keyframes(media_files, spec, tmp_path):
    path = media_files[spec.name]
    keyframes = get_keyframe_timestamps_pyav(str(path))
    for i, (a, b) in enumerate(spec.scene_frames):
        clip, mode, offset = cut_scene(
            path, spec.frame_time(a), spec.frame_time(b), i, tmp_path, keyframes, "copy"
        )
        assert mode == "copy"
        expected = spec.copy_frames(a, b)
        assert [f.number for f in read_frames(clip)] == list(expected), f"scene {i}"
        assert offset == pytest.approx(spec.frame_time(a) - spec.frame_time(expected.start), abs=1e-4)
        assert video_stream(clip)["codec_name"] == video_stream(path)["codec_name"]
        assert Path(clip).suffix == (".mov" if spec.codec == "prores" else ".mp4")


@pytest.mark.media
@pytest.mark.parametrize("spec", media_params("h264_2997_gop", "h265_60_main10_gop", "h264_25_sparse"))
def test_copy_poster_offset_skips_bleed(media_files, spec, tmp_path):
    path = media_files[spec.name]
    keyframes = get_keyframe_timestamps_pyav(str(path))
    for i, (a, b) in enumerate(spec.scene_frames):
        clip, _, offset = cut_scene(path, spec.frame_time(a), spec.frame_time(b), i, tmp_path, keyframes, "copy")
        thumb = str(tmp_path / f"poster_{i}.jpg")
        assert make_thumbnail(clip, thumb, seek_sec=offset)
        assert jpeg_color(thumb) == spec.segment_color(i), f"scene {i}"


@pytest.mark.media
def test_cut_all_scenes_reports_every_scene(media_files, tmp_path):
    spec = BY_NAME["h265_25_cuts"]
    path = media_files[spec.name]
    scenes = [
        {"scene_index": i, "start_sec": s, "end_sec": e}
        for i, (s, e) in enumerate(spec.scene_secs)
    ]
    seen = []
    results = cut_all_scenes(
        path, scenes, get_keyframe_timestamps_pyav(str(path)), tmp_path, "copy",
        max_workers=3, on_ready=seen.append, emit_progress_updates=False,
    )
    assert sorted(r["scene_index"] for r in results) == list(range(len(scenes)))
    assert len(seen) == len(scenes)
    assert all(r["clip_mode"] == "copy" and os.path.exists(r["clip_path"]) for r in results)


@pytest.mark.media
def test_cut_scene_rejects_bad_input(media_files, tmp_path):
    path = media_files["h264_24_cuts"]
    with pytest.raises(ValueError):
        cut_scene(path, 2.0, 2.0, 0, tmp_path, [0.0], "reencode")
    with pytest.raises(ValueError):
        cut_scene(path, 0.0, 1.0, 0, tmp_path, [0.0], "exact")


@pytest.mark.media
@pytest.mark.parametrize("spec", media_params())
def test_segmenter_tiles_every_frame_once(media_files, spec, tmp_path):
    path = str(media_files[spec.name])
    cuts = spec.cut_secs
    run_ffmpeg_segment(path, str(tmp_path / "seg_%04d.mp4"), cuts)
    scenes = collect_scenes(str(tmp_path), "seg", cuts, spec.duration)
    numbers = [n for s in scenes for n in (f.number for f in read_frames(s["path"]))]
    assert numbers == list(range(spec.total_frames))


@pytest.mark.media
@pytest.mark.parametrize("spec", media_params("h264_24_cuts", "h265_25_cuts", "prores_50_4444"))
def test_segmenter_on_keyframe_cuts_is_exact(media_files, spec, tmp_path):
    path = str(media_files[spec.name])
    run_ffmpeg_segment(path, str(tmp_path / "seg_%04d.mp4"), spec.cut_secs)
    scenes = collect_scenes(str(tmp_path), "seg", spec.cut_secs, spec.duration)
    assert len(scenes) == len(spec.segments)
    for i, scene in enumerate(scenes):
        a, b = spec.scene_frames[i]
        assert [f.number for f in read_frames(scene["path"])] == list(range(a, b))


class TestShortSceneMerge:
    def _scenes(self, bounds):
        return [
            {"scene_index": i, "start_sec": a, "end_sec": b, "duration_sec": b - a}
            for i, (a, b) in enumerate(zip(bounds, bounds[1:]))
        ]

    @pytest.mark.media
    def test_short_scene_folds_into_matching_previous(self, media_files):
        spec = BY_NAME["h264_24_cuts"]
        scenes = self._scenes([0.0, 1.9, 2.0, 3.5])
        merged = merge_short_scenes(scenes, str(media_files[spec.name]))
        assert [(s["start_sec"], s["end_sec"]) for s in merged] == [(0.0, 2.0), (2.0, 3.5)]
        assert [s["scene_index"] for s in merged] == [0, 1]

    @pytest.mark.media
    def test_short_scene_folds_into_matching_next(self, media_files):
        spec = BY_NAME["h264_24_cuts"]
        scenes = self._scenes([0.0, 2.0, 2.1, 3.5])
        merged = merge_short_scenes(scenes, str(media_files[spec.name]))
        assert [(s["start_sec"], s["end_sec"]) for s in merged] == [(0.0, 2.0), (2.0, 3.5)]

    @pytest.mark.media
    def test_real_three_frame_scene_is_merged(self, media_files):
        spec = BY_NAME["h264_30_short"]
        merged = merge_short_scenes(self._scenes([0.0, *spec.cut_secs, spec.duration]), str(media_files[spec.name]))
        assert len(merged) == len(spec.segments) - 1
        assert all(s["duration_sec"] >= 0.25 for s in merged)
        assert merged[0]["start_sec"] == 0.0 and merged[-1]["end_sec"] == spec.duration

    def test_nothing_short_is_untouched(self):
        scenes = self._scenes([0.0, 1.0, 2.0])
        assert merge_short_scenes(scenes, "unused.mp4") is scenes
