from __future__ import annotations

import numpy as np
import pytest


try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


@pytest.fixture
def detector():
    if not HAS_CV2:
        pytest.skip("cv2 not installed")
    from amverge.core.deadframes.engine import DeadFrameDetector

    return DeadFrameDetector()


def _make_gray(rows, cols, value=128):
    return np.full((rows, cols), value, dtype=np.uint8)


def _add_rect(gray, x, y, w, h, value=200):
    gray[y : y + h, x : x + w] = value


class TestDetection:
    def test_static_frames_detected_dead(self, detector):
        a = _make_gray(64, 64, 128)
        b = _make_gray(64, 64, 128)
        assert detector.is_deadframe(a, b)

    def test_moving_subject_kept(self, detector):
        detector.flow_threshold = 0.01
        detector.motion_area_fraction = 0.01
        a = _make_gray(64, 64, 128)
        _add_rect(a, 10, 10, 10, 10, 220)
        b = _make_gray(64, 64, 128)
        _add_rect(b, 20, 20, 10, 10, 220)
        assert not detector.is_deadframe(a, b)

    def test_camera_pan_dead_default(self, detector):
        a = _make_gray(64, 64, 128)
        _add_rect(a, 5, 5, 8, 8, 220)
        b = _make_gray(64, 64, 128)
        _add_rect(b, 13, 13, 8, 8, 220)
        detector.skip_homography = False
        detector.parallax_mode = False
        detector.homography_inlier_ratio = 0.1
        result = detector.is_deadframe(a, b)
        assert result

    def test_camera_pan_kept_with_keep_camera(self, detector):
        detector.flow_threshold = 0.01
        detector.motion_area_fraction = 0.01
        a = _make_gray(64, 64, 128)
        _add_rect(a, 5, 5, 8, 8, 220)
        b = _make_gray(64, 64, 128)
        _add_rect(b, 13, 13, 8, 8, 220)
        detector.skip_homography = True
        assert not detector.is_deadframe(a, b)

    def test_talking_head_keep_talking(self, detector):
        detector.flow_threshold = 0.05
        detector.motion_area_fraction = 0.0
        a = _make_gray(64, 64, 128)
        _add_rect(a, 10, 10, 15, 15, 220)
        b = _make_gray(64, 64, 128)
        _add_rect(b, 13, 13, 15, 15, 220)
        assert not detector.is_deadframe(a, b)

    def test_camera_shake_kept(self, detector):
        detector.flow_threshold = 0.01
        detector.motion_area_fraction = 0.01
        a = _make_gray(64, 64, 128)
        _add_rect(a, 10, 10, 20, 20, 220)
        b = _make_gray(64, 64, 128)
        _add_rect(b, 8, 9, 20, 20, 220)
        detector.skip_homography = False
        detector.homography_inlier_ratio = 0.9
        result = detector.is_deadframe(a, b)
        assert not result

    def test_transient_passer_still_dead(self, detector):
        a = _make_gray(64, 64, 128)
        b = _make_gray(64, 64, 128)
        _add_rect(b, 60, 60, 2, 2, 250)
        detector.motion_area_fraction = 0.01
        result = detector.is_deadframe(a, b)
        assert result

    def test_brightness_flash_dead(self, detector):
        a = _make_gray(64, 64, 50)
        b = _make_gray(64, 64, 200)
        assert detector.is_deadframe(a, b)

    def test_flat_frame_dead(self, detector):
        a = _make_gray(64, 64, 128)
        b = _make_gray(64, 64, 128)
        assert detector.is_deadframe(a, b)


class TestCadenceSmoothing:
    def test_run_shorter_than_cadence_reverted(self):
        from amverge.core.deadframes.engine import _smooth_decisions

        decisions = [True, False, False, True, True, True, True]
        smoothed = _smooth_decisions(decisions, min_dead=3)
        assert smoothed == [True, True, True, True, True, True, True]

    def test_run_equal_to_cadence_unaffected(self):
        from amverge.core.deadframes.engine import _smooth_decisions

        decisions = [True, False, False, False, True]
        smoothed = _smooth_decisions(decisions, min_dead=3)
        assert smoothed == [True, False, False, False, True]

    def test_run_longer_than_cadence_unaffected(self):
        from amverge.core.deadframes.engine import _smooth_decisions

        decisions = [True, False, False, False, False, True]
        smoothed = _smooth_decisions(decisions, min_dead=3)
        assert smoothed == [True, False, False, False, False, True]


class TestSegmentation:
    def test_keep_runs_to_segments(self):
        from amverge.core.deadframes.engine import _decisions_to_segments

        decisions = [True, True, False, False, True, True, True]
        segments = _decisions_to_segments(decisions, fps=10.0)
        assert len(segments) == 2
        assert segments[0] == (0.0, 0.2)
        assert segments[1] == (0.4, 0.7)

    def test_all_keep_single_segment(self):
        from amverge.core.deadframes.engine import _decisions_to_segments

        decisions = [True, True, True]
        segments = _decisions_to_segments(decisions, fps=10.0)
        assert len(segments) == 1
        assert segments[0] == (0.0, 0.3)


class TestAutoCalibration:
    def test_thresholds_in_clamp_bounds(self, detector):
        if not HAS_CV2:
            pytest.skip("cv2 not installed")
        stats = []
        stats.append(
            {
                "mean_mag": 1.0,
                "diff_fraction": 0.2,
                "inlier_ratio": 0.0,
                "frob_norm": None,
                "is_dead": False,
            }
        )
        stats.append(
            {
                "mean_mag": 2.0,
                "diff_fraction": 0.3,
                "inlier_ratio": 0.0,
                "frob_norm": None,
                "is_dead": False,
            }
        )
        stats.append(
            {
                "mean_mag": 0.5,
                "diff_fraction": 0.1,
                "inlier_ratio": 0.0,
                "frob_norm": None,
                "is_dead": False,
            }
        )
        stats.append(
            {
                "mean_mag": 3.0,
                "diff_fraction": 0.5,
                "inlier_ratio": 0.0,
                "frob_norm": None,
                "is_dead": False,
            }
        )
        detector._compute_thresholds_from_stats(stats)
        assert 0.2 <= detector.flow_threshold <= 2.0
        assert 0.03 <= detector.motion_area_fraction <= 0.25


class TestRegistry:
    def test_registry_has_heuristic(self):
        from amverge.core.deadframes.registry import DEADFRAMES_REGISTRY, get_model

        assert "heuristic" in DEADFRAMES_REGISTRY
        entry = get_model("heuristic")
        assert entry is not None
        assert entry["method"] == "heuristic"

    def test_get_unknown_model_returns_none(self):
        from amverge.core.deadframes.registry import get_model

        assert get_model("nonexistent") is None

    def test_all_keys_nonempty(self):
        from amverge.core.deadframes.registry import get_all_model_keys

        keys = get_all_model_keys()
        assert len(keys) > 0
        assert "heuristic" in keys


class TestEngineAvailability:
    def test_deadframes_available(self):
        from amverge.core.deadframes import DEADFRAMES_AVAILABLE

        assert DEADFRAMES_AVAILABLE == HAS_CV2


class TestBuildFfmpegCmd:
    def test_h264_output_cfr(self):
        from amverge.core.deadframes.engine import _build_ffmpeg_cmd

        segments = [(0.0, 0.5), (1.0, 2.0)]
        cmd = _build_ffmpeg_cmd(
            input_path="test.mp4",
            output_path="out.mp4",
            segments=segments,
            fps=30.0,
            pix_fmt="yuv420p",
            prores=False,
            no_audio=True,
        )
        assert cmd[0].endswith("ffmpeg") or "ffmpeg" in cmd[0]
        assert "-fps_mode" in cmd
        cfr_idx = cmd.index("-fps_mode")
        assert cmd[cfr_idx + 1] == "cfr"
        assert cmd[cfr_idx + 2] == "-r"
        assert cmd[cfr_idx + 3] == "30.0"
        assert "-c:v" in cmd
        vcodec_idx = cmd.index("-c:v")
        assert cmd[vcodec_idx + 1] == "libx264"
        assert "out.mp4" == cmd[-1]

    def test_10bit_uses_hevc(self):
        from amverge.core.deadframes.engine import _build_ffmpeg_cmd

        segments = [(0.0, 1.0)]
        cmd = _build_ffmpeg_cmd(
            input_path="test.mp4",
            output_path="out.mp4",
            segments=segments,
            fps=24.0,
            pix_fmt="yuv420p10le",
            prores=False,
            no_audio=True,
        )
        assert "-c:v" in cmd
        vcodec_idx = cmd.index("-c:v")
        assert cmd[vcodec_idx + 1] == "libx265"

    def test_prores_encoder(self):
        from amverge.core.deadframes.engine import _build_ffmpeg_cmd

        segments = [(0.0, 1.0)]
        cmd = _build_ffmpeg_cmd(
            input_path="test.mp4",
            output_path="out.mov",
            segments=segments,
            fps=30.0,
            pix_fmt="yuv420p",
            prores=True,
            no_audio=True,
        )
        assert "-c:v" in cmd
        vcodec_idx = cmd.index("-c:v")
        assert cmd[vcodec_idx + 1] == "prores_ks"
