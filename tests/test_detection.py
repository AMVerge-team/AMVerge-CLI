from __future__ import annotations

import json
import os
import subprocess
import sys

import numpy as np
import pytest

from amverge import detect_scenes
from tests.fixtures import BY_NAME, known_bug, media_params
from tests.fixtures.known_bugs import TRANSNET_MPS
from tests.fixtures.media import jpeg_color, read_frames

pytestmark = pytest.mark.media


def _clip_numbers(result) -> list[list[int]]:
    return [[f.number for f in read_frames(s.path)] for s in result.scenes]


def _assert_contiguous(result, spec) -> None:
    assert result.scenes[0].start == 0.0
    assert result.scenes[-1].end == pytest.approx(spec.duration, abs=0.05)
    for prev, cur in zip(result.scenes, result.scenes[1:]):
        assert cur.start == pytest.approx(prev.end)
    assert [s.index for s in result.scenes] == list(range(len(result.scenes)))


class TestKeyframeMethod:
    @pytest.mark.parametrize("spec", media_params())
    def test_clips_tile_the_source(self, media_files, spec, tmp_path):
        result = detect_scenes(
            str(media_files[spec.name]), str(tmp_path), method="keyframe",
            thumbnails=False, similarity=False,
        )
        _assert_contiguous(result, spec)
        numbers = [n for clip in _clip_numbers(result) for n in clip]
        assert numbers == list(range(spec.total_frames))

    @pytest.mark.parametrize("spec", media_params("h264_24_cuts", "h265_25_cuts"))
    def test_aligned_source_gives_one_clip_per_color(self, media_files, spec, tmp_path):
        result = detect_scenes(
            str(media_files[spec.name]), str(tmp_path), method="keyframe", similarity=False,
        )
        assert [(s.start, s.end) for s in result.scenes] == pytest.approx(spec.scene_secs, abs=1e-3)
        for i, clip in enumerate(result.scenes):
            a, b = spec.scene_frames[i]
            frames = read_frames(clip.path)
            assert [f.number for f in frames] == list(range(a, b))
            assert {f.color for f in frames} == {spec.segment_color(i)}
            assert os.path.exists(clip.thumbnail)

    @pytest.mark.parametrize("spec", media_params("prores_23976_422hq", "prores_50_4444"))
    def test_intra_source_is_split_every_min_duration(self, media_files, spec, tmp_path):
        result = detect_scenes(
            str(media_files[spec.name]), str(tmp_path), method="keyframe",
            min_duration=0.5, thumbnails=False, similarity=False,
        )
        assert len(result.scenes) >= int(spec.duration / 0.5) - 1
        for clip, numbers in zip(result.scenes, _clip_numbers(result)):
            assert numbers, f"scene {clip.index} is empty"
            assert numbers[0] == spec.frame_at(clip.start)

    def test_min_duration_folds_short_scene(self, media_files, tmp_path):
        spec = BY_NAME["h264_30_short"]
        path = str(media_files[spec.name])
        fine = detect_scenes(path, str(tmp_path / "a"), min_duration=0.0, thumbnails=False, similarity=False)
        coarse = detect_scenes(path, str(tmp_path / "b"), min_duration=0.25, thumbnails=False, similarity=False)
        assert len(fine.scenes) == 4
        assert len(coarse.scenes) == 3
        assert all(s.duration >= 0.25 for s in coarse.scenes)

    def test_similarity_flags_same_color_neighbours(self, media_files, tmp_path):
        spec = BY_NAME["h264_2997_gop"]
        result = detect_scenes(str(media_files[spec.name]), str(tmp_path), method="keyframe", min_duration=0.0)
        first_colors = [read_frames(s.path)[0].color for s in result.scenes]
        same = {
            (i, i + 1) for i in range(len(first_colors) - 1)
            if first_colors[i] == first_colors[i + 1]
        }
        assert len(same) >= 3, "fixture should produce same-color neighbours"
        assert set(result.similar_pairs) <= same
        assert len(result.similar_pairs) >= len(same) - 1

    def test_aligned_cuts_have_no_similar_pairs(self, media_files, tmp_path):
        result = detect_scenes(str(media_files["h264_24_cuts"]), str(tmp_path))
        assert result.similar_pairs == []

    def test_scenes_json_matches_result(self, media_files, tmp_path):
        result = detect_scenes(str(media_files["h265_25_cuts"]), str(tmp_path), similarity=False)
        payload = json.loads(open(result.scenes_json).read())
        assert payload["scenes"] == [s.to_dict() for s in result.scenes]

    def test_progress_reports_every_stage(self, media_files, tmp_path):
        stages = []
        detect_scenes(str(media_files["h264_24_cuts"]), str(tmp_path), progress=lambda s, p, m: stages.append(s))
        assert {"detect", "segment", "thumbnails", "similarity"} <= set(stages)


class TestEdgeMethod:
    @pytest.fixture(autouse=True)
    def _cv2(self):
        pytest.importorskip("cv2")

    @pytest.mark.parametrize("spec", media_params(
        "h264_24_cuts", "h264_2997_gop", "h265_60_main10_gop",
    ))
    def test_edge_cuts_stay_in_keyframe_windows(self, media_files, spec, tmp_path):
        radius = 0.6
        result = detect_scenes(
            str(media_files[spec.name]), str(tmp_path), method="edge",
            edge_radius=radius, min_duration=0.25, thumbnails=False, similarity=False,
        )
        _assert_contiguous(result, spec)
        for scene in result.scenes[1:]:
            assert min(abs(scene.start - k) for k in spec.keyframe_secs) <= radius + 1e-3
        numbers = [n for clip in _clip_numbers(result) for n in clip]
        assert numbers == list(range(spec.total_frames))


@pytest.mark.ml
@pytest.mark.slow
class TestTransNetV2Method:
    @pytest.mark.parametrize("spec", media_params(
        "h264_24_cuts", "h264_2997_gop", "h265_60_main10_gop", "prores_23976_422hq",
    ))
    def test_cuts_land_on_color_changes_and_clips_are_exact(self, media_files, spec, tmp_path, transnet):
        result = detect_scenes(
            str(media_files[spec.name]), str(tmp_path), method="transnetv2",
            thumbnails=False, similarity=False,
        )
        starts = [spec.frame_at(s.start) for s in result.scenes]
        assert starts == [0, *spec.cut_frames]
        for i, clip in enumerate(result.scenes):
            a, b = spec.scene_frames[i]
            frames = read_frames(clip.path)
            assert [f.number for f in frames] == list(range(a, b)), f"scene {i}"
            assert {f.color for f in frames} == {spec.segment_color(i)}

    def test_nelux_request_falls_back_to_ffmpeg(self, media_files, tmp_path, transnet):
        from amverge.core.detection.nelux_runtime import nelux_available
        if nelux_available():
            pytest.skip("nelux is installed here; fallback path not exercised")
        path = str(media_files["h265_25_cuts"])
        messages = []
        a = detect_scenes(path, str(tmp_path / "a"), method="transnetv2", thumbnails=False, similarity=False)
        b = detect_scenes(
            path, str(tmp_path / "b"), method="transnetv2", decode_method="nelux",
            thumbnails=False, similarity=False, progress=lambda s, p, m: messages.append(m),
        )
        assert [(s.start, s.end) for s in a.scenes] == [(s.start, s.end) for s in b.scenes]
        assert any("falling back" in m.lower() for m in messages)

    def test_thumbnails_show_the_scene_color(self, media_files, tmp_path, transnet):
        spec = BY_NAME["h264_2997_gop"]
        result = detect_scenes(str(media_files[spec.name]), str(tmp_path), method="transnetv2", similarity=False)
        assert [jpeg_color(s.thumbnail) for s in result.scenes] == [
            spec.segment_color(i) for i in range(len(spec.segments))
        ]


@pytest.mark.ml
@known_bug(TRANSNET_MPS)
def test_transnet_runs_on_apple_silicon_when_torch_imported_first(media_files, tmp_path, transnet):
    import torch
    if not torch.backends.mps.is_available():
        pytest.skip("no MPS device")
    env = {k: v for k, v in os.environ.items() if k != "PYTORCH_ENABLE_MPS_FALLBACK"}
    script = (
        "import sys, torch; from amverge import detect_scenes; "
        "detect_scenes(sys.argv[1], sys.argv[2], method='transnetv2', thumbnails=False, similarity=False)"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script, str(media_files["h264_24_cuts"]), str(tmp_path)],
        env=env, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]


def test_transnet_seconds_are_contiguous_and_exclusive():
    from amverge.core.video.scene_utils import transnet_scenes_to_seconds
    secs = transnet_scenes_to_seconds(np.array([[0, 89], [90, 239], [240, 389]]), 60.0)
    assert secs.tolist() == [[0.0, 1.5], [1.5, 4.0], [4.0, 6.5]]
