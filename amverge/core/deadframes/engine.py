from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

try:
    import cv2

    DEADFRAMES_AVAILABLE = True
except ImportError:
    cv2 = None
    DEADFRAMES_AVAILABLE = False

try:
    import numpy as _np
except ImportError:
    _np = None

from ..infra.binaries import get_ffmpeg, get_ffprobe

_CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def _get_video_info(input_path: str) -> dict[str, Any]:
    ffprobe = get_ffprobe()
    cmd = [
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,width,height,pix_fmt,field_order,codec_name,duration",
        "-of", "json",
        input_path,
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True,
        creationflags=_CREATE_NO_WINDOW,
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFprobe failed: {result.stderr.strip()}")
    import json

    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError(f"No video stream found in: {input_path}")
    s = streams[0]
    rfr = s.get("r_frame_rate", "30/1")

    def _parse_fraction(frac_str: str) -> float:
        if "/" in frac_str:
            num, den = frac_str.split("/")
            return float(num) / float(den)
        return float(frac_str)

    fps = _parse_fraction(rfr)
    duration = float(s.get("duration", 0))
    if duration <= 0:
        w = s.get("width", 0)
        h = s.get("height", 0)
        if w > 0 and h > 0:
            total = s.get("nb_frames")
            if total:
                duration = float(total) / fps
    return {
        "fps": fps,
        "width": s.get("width", 0),
        "height": s.get("height", 0),
        "pix_fmt": s.get("pix_fmt", "yuv420p"),
        "duration": duration,
    }


def _get_audio_info(input_path: str) -> Optional[dict[str, Any]]:
    ffprobe = get_ffprobe()
    try:
        cmd = [
            ffprobe,
            "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=codec_name,sample_rate,channels",
            "-of", "json",
            input_path,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            creationflags=_CREATE_NO_WINDOW,
        )
        if result.returncode != 0:
            return None
        import json

        data = json.loads(result.stdout)
        streams = data.get("streams", [])
        if streams:
            return streams[0]
        return None
    except Exception:
        return None


def _decisions_to_segments(
    decisions: list[bool], fps: float
) -> list[tuple[float, float]]:
    segments: list[tuple[float, float]] = []
    in_segment = False
    segment_start = 0.0
    for i, keep in enumerate(decisions):
        t = i / fps
        if keep and not in_segment:
            segment_start = t
            in_segment = True
        elif not keep and in_segment:
            segments.append((segment_start, t))
            in_segment = False
    if in_segment:
        segments.append((segment_start, len(decisions) / fps))
    return segments


def _smooth_decisions(decisions: list[bool], min_dead: int) -> list[bool]:
    if min_dead <= 0:
        return decisions
    smoothed = list(decisions)
    n = len(smoothed)
    i = 0
    while i < n:
        if not smoothed[i]:
            start = i
            while i < n and not smoothed[i]:
                i += 1
            if (i - start) < min_dead:
                for j in range(start, i):
                    smoothed[j] = True
        else:
            i += 1
    return smoothed


def _get_color_args(input_path: str) -> list[str]:
    ffprobe = get_ffprobe()
    try:
        out = subprocess.check_output(
            [
                ffprobe, "-v", "error", "-select_streams", "v:0",
                "-show_entries",
                "stream=color_primaries,color_transfer,color_space,color_range",
                "-of", "default=nw=1", str(input_path),
            ],
            text=True, stderr=subprocess.DEVNULL,
            creationflags=_CREATE_NO_WINDOW,
        )
    except Exception:
        return []
    vals = {}
    for line in out.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            vals[k] = v.strip()

    def ok(x):
        return x and x.lower() not in ("unknown", "n/a", "reserved", "")

    args = []
    if ok(vals.get("color_primaries")):
        args += ["-color_primaries", vals["color_primaries"]]
    if ok(vals.get("color_transfer")):
        args += ["-color_trc", vals["color_transfer"]]
    if ok(vals.get("color_space")):
        args += ["-colorspace", vals["color_space"]]
    if ok(vals.get("color_range")):
        args += ["-color_range", vals["color_range"]]
    return args


def _build_ffmpeg_cmd(
    input_path: str,
    output_path: str,
    segments: list[tuple[float, float]],
    fps: float,
    pix_fmt: str,
    prores: bool = False,
    no_audio: bool = False,
    audio_filter: str = "",
) -> list[str]:
    filter_parts: list[str] = []
    video_labels: list[str] = []
    for i, (start, end) in enumerate(segments):
        filter_parts.append(
            f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{i}]"
        )
        video_labels.append(f"[v{i}]")
    video_concat = "".join(video_labels)
    filter_parts.append(
        f"{video_concat}concat=n={len(segments)}:v=1:a=0[outv]"
    )

    has_audio_track = bool(audio_filter)
    if has_audio_track:
        filter_parts.append(audio_filter)

    is_10bit = "10" in pix_fmt or "p10" in pix_fmt
    color_args = _get_color_args(input_path)

    if prores:
        cmd = [
            get_ffmpeg(), "-y",
            "-fflags", "+genpts",
            "-i", input_path,
            "-filter_complex", ";".join(filter_parts),
            "-map", "[outv]",
            "-c:v", "prores_ks",
            "-profile:v", "3",
            "-vendor", "ap10",
            "-pix_fmt", "yuv422p10le",
            "-fps_mode", "cfr",
            "-r", str(fps),
        ]
        if has_audio_track:
            cmd += ["-map", "[outa]"]
        else:
            cmd += ["-an"]
        if color_args:
            cmd += color_args
        cmd.append(output_path)
        return cmd

    encoder = "libx265" if is_10bit else "libx264"
    output_pix_fmt = pix_fmt if is_10bit else "yuv420p"

    cmd = [
        get_ffmpeg(), "-y",
        "-fflags", "+genpts",
        "-i", input_path,
        "-filter_complex", ";".join(filter_parts),
        "-map", "[outv]",
        "-fps_mode", "cfr",
        "-r", str(fps),
        "-c:v", encoder,
        "-pix_fmt", output_pix_fmt,
        "-crf", "18",
        "-preset", "medium",
        "-tune", "animation",
        "-profile:v", "high",
        "-movflags", "+faststart",
    ]

    if has_audio_track:
        cmd += ["-map", "[outa]"]
    elif no_audio:
        cmd += ["-an"]

    if is_10bit:
        # 10-bit encodes with libx265, whose profile is main10; the h264 "high"
        # profile is invalid here and left ffmpeg treating "high" as an output
        # filename ("Unable to choose an output format for 'high'").
        cmd[cmd.index("high")] = "main10"

    if color_args:
        cmd += color_args

    cmd.append(output_path)
    return cmd


def _build_audio_filter(
    segments: list[tuple[float, float]],
) -> str:
    audio_parts: list[str] = []
    audio_labels: list[str] = []
    for i, (start, end) in enumerate(segments):
        audio_parts.append(
            f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{i}]"
        )
        audio_labels.append(f"[a{i}]")
    audio_concat = "".join(audio_labels)
    audio_parts.append(
        f"{audio_concat}concat=n={len(segments)}:v=0:a=1[outa]"
    )
    return ";".join(audio_parts)


class DeadFrameDetector:
    def __init__(
        self,
        flow_threshold: float = 0.5,
        motion_area_fraction: float = 0.15,
        homography_inlier_ratio: float = 0.5,
        orb_features: int = 500,
        min_matches: int = 8,
        ransac_threshold: float = 5.0,
        detect_scale: float = 1.0,
        keep_talking: bool = False,
        keep_camera: bool = False,
        parallax_mode: bool = False,
    ):
        if cv2 is None:
            raise ImportError(
                "OpenCV (cv2) is required. Install with: pip install amverge[deadframes]"
            )
        self.cv2 = cv2
        self.flow_threshold = 0.05 if keep_talking else flow_threshold
        self.motion_area_fraction = 0.0 if keep_talking else motion_area_fraction
        self.homography_inlier_ratio = homography_inlier_ratio
        self.skip_homography = keep_camera
        self.parallax_mode = parallax_mode
        self.orb_features = orb_features
        self.min_matches = min_matches
        self.ransac_threshold = ransac_threshold
        self.detect_scale = max(0.1, min(1.0, detect_scale))
        self.orb = self.cv2.ORB_create(nfeatures=orb_features)
        self.matcher = self.cv2.BFMatcher(self.cv2.NORM_HAMMING, crossCheck=True)

    def _maybe_scale(self, gray):
        if self.detect_scale >= 1.0:
            return gray, 1.0
        h, w = gray.shape
        nh, nw = int(h * self.detect_scale), int(w * self.detect_scale)
        return self.cv2.resize(gray, (nw, nh)), self.detect_scale

    def _collect_pair_stats(
        self, prev_gray, curr_gray
    ):
        h, w = curr_gray.shape
        mean_prev = float(self.cv2.mean(prev_gray)[0])
        mean_curr = float(self.cv2.mean(curr_gray)[0])
        if abs(mean_curr - mean_prev) > 40:
            return 0.0, 0.0, 0.0, None
        var_curr = float(self.cv2.meanStdDev(curr_gray)[1][0][0]) ** 2
        if var_curr < 5.0:
            return 0.0, 0.0, 0.0, None
        flow = self.cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, _ = self.cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_mag = float(self.cv2.mean(mag)[0])
        diff_frame = self.cv2.absdiff(prev_gray, curr_gray)
        _, diff_mask = self.cv2.threshold(
            diff_frame, 25, 255, self.cv2.THRESH_BINARY
        )
        diff_mask = diff_mask.astype("uint8")
        kernel = self.cv2.getStructuringElement(self.cv2.MORPH_ELLIPSE, (5, 5))
        diff_mask = self.cv2.morphologyEx(diff_mask, self.cv2.MORPH_CLOSE, kernel)
        contours, _ = self.cv2.findContours(
            diff_mask, self.cv2.RETR_EXTERNAL, self.cv2.CHAIN_APPROX_SIMPLE
        )
        total_area = float(h * w)
        diff_area = 0.0
        for c in contours:
            area = self.cv2.contourArea(c)
            if area > 25:
                diff_area += area
        diff_fraction = diff_area / total_area
        inlier_ratio = 0.0
        frob_norm = None
        kp1, des1 = self.orb.detectAndCompute(prev_gray, None)
        kp2, des2 = self.orb.detectAndCompute(curr_gray, None)
        if des1 is not None and des2 is not None and len(des1) >= self.min_matches:
            try:
                matches = self.matcher.match(des1, des2)
                matches = sorted(matches, key=lambda m: m.distance)
                good = matches[: min(200, len(matches))]
                if len(good) >= self.min_matches:
                    src_pts = self.cv2.KeyPoint_convert(
                        [kp1[m.queryIdx] for m in good]
                    )
                    dst_pts = self.cv2.KeyPoint_convert(
                        [kp2[m.trainIdx] for m in good]
                    )
                    H, mask = self.cv2.findHomography(
                        src_pts, dst_pts, self.cv2.RANSAC, self.ransac_threshold,
                        maxIters=500,
                    )
                    if H is not None and mask is not None:
                        inlier_count = int(self.cv2.countNonZero(mask))
                        inlier_ratio = inlier_count / len(good)
                        if _np is not None:
                            frob_norm = float(
                                _np.linalg.norm(H.astype(_np.float64) - _np.eye(3))
                            )
            except self.cv2.error:
                pass
        return mean_mag, diff_fraction, inlier_ratio, frob_norm

    def is_deadframe(self, prev_gray, curr_gray) -> bool:
        h, w = curr_gray.shape
        mean_prev = float(self.cv2.mean(prev_gray)[0])
        mean_curr = float(self.cv2.mean(curr_gray)[0])
        if abs(mean_curr - mean_prev) > 40:
            return True
        var_curr = float(self.cv2.meanStdDev(curr_gray)[1][0][0]) ** 2
        if var_curr < 5.0:
            return True
        flow = self.cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, _ = self.cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_mag = float(self.cv2.mean(mag)[0])
        if mean_mag < self.flow_threshold:
            return True
        kp1, des1 = self.orb.detectAndCompute(prev_gray, None)
        kp2, des2 = self.orb.detectAndCompute(curr_gray, None)
        if des1 is not None and des2 is not None and len(des1) >= self.min_matches:
            try:
                matches = self.matcher.match(des1, des2)
                matches = sorted(matches, key=lambda m: m.distance)
                good = matches[: min(200, len(matches))]
                if len(good) >= self.min_matches:
                    src_pts = self.cv2.KeyPoint_convert(
                        [kp1[m.queryIdx] for m in good]
                    )
                    dst_pts = self.cv2.KeyPoint_convert(
                        [kp2[m.trainIdx] for m in good]
                    )
                    H, mask = self.cv2.findHomography(
                        src_pts, dst_pts, self.cv2.RANSAC, self.ransac_threshold,
                        maxIters=500,
                    )
                    if H is not None and mask is not None:
                        inlier_count = int(self.cv2.countNonZero(mask))
                        inlier_ratio = inlier_count / len(good)
                        if (
                            not self.skip_homography
                            and inlier_ratio > self.homography_inlier_ratio
                        ):
                            if _np is not None:
                                frob_norm = _np.linalg.norm(
                                    H.astype(_np.float64) - _np.eye(3)
                                )
                                if frob_norm > 1.0:
                                    return not self.parallax_mode
            except self.cv2.error:
                pass
        diff_frame = self.cv2.absdiff(prev_gray, curr_gray)
        _, diff_mask = self.cv2.threshold(
            diff_frame, 25, 255, self.cv2.THRESH_BINARY
        )
        diff_mask = diff_mask.astype("uint8")
        kernel = self.cv2.getStructuringElement(self.cv2.MORPH_ELLIPSE, (5, 5))
        diff_mask = self.cv2.morphologyEx(
            diff_mask, self.cv2.MORPH_CLOSE, kernel
        )
        contours, _ = self.cv2.findContours(
            diff_mask, self.cv2.RETR_EXTERNAL, self.cv2.CHAIN_APPROX_SIMPLE
        )
        total_area = float(h * w)
        diff_area = 0.0
        for c in contours:
            area = self.cv2.contourArea(c)
            if area > 25:
                diff_area += area
        diff_fraction = diff_area / total_area
        return diff_fraction < self.motion_area_fraction

    def _is_dead_from_stats(
        self,
        mean_mag: float,
        diff_frac: float,
        inlier_ratio: float,
        frob_norm: Optional[float],
    ) -> bool:
        if mean_mag < self.flow_threshold:
            return True
        if not self.skip_homography and (
            inlier_ratio > self.homography_inlier_ratio
            and frob_norm is not None
            and frob_norm > 1.0
        ):
            return not self.parallax_mode
        if diff_frac < self.motion_area_fraction:
            return True
        return False

    def _compute_thresholds_from_stats(
        self, stats_list: list[dict[str, Optional[float]]],
        lock_flow: bool = False,
        lock_area: bool = False,
    ) -> None:
        if _np is None:
            return
        if not lock_flow:
            flow_means = [
                float(s["mean_mag"]) for s in stats_list if s["mean_mag"] is not None
            ]
            if len(flow_means) >= 2:
                p_flow = float(_np.percentile(flow_means, 20))
                self.flow_threshold = max(0.2, min(2.0, p_flow * 0.6))
        if not lock_area:
            diff_fracs = [
                float(s["diff_fraction"])
                for s in stats_list
                if s["diff_fraction"] is not None
            ]
            if len(diff_fracs) >= 2:
                p_diff = float(_np.percentile(diff_fracs, 15))
                self.motion_area_fraction = max(0.03, min(0.25, p_diff * 0.4))

    def process_video(
        self,
        input_path: str,
        auto: bool = False,
        auto_sample_limit: int = 0,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        preview_cb: Optional[Callable[[object, int], None]] = None,
    ) -> list[bool]:
        cap = self.cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {input_path}")
        total_frames = int(cap.get(self.cv2.CAP_PROP_FRAME_COUNT))
        stats_list: list[dict[str, object]] = []
        decisions: list[bool] = [True]
        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            return decisions
        prev_gray = self.cv2.cvtColor(prev_frame, self.cv2.COLOR_BGR2GRAY)
        prev_gray, _ = self._maybe_scale(prev_gray)
        pair_idx = 0
        start_time = time.time()

        def _emit(pct, stage="Detecting"):
            if progress_cb:
                progress_cb(pct, f"{stage}... frame {pair_idx}/{total_frames - 1}")

        _emit(0)

        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break
            curr_gray = self.cv2.cvtColor(curr_frame, self.cv2.COLOR_BGR2GRAY)
            curr_gray, _ = self._maybe_scale(curr_gray)
            try:
                mean_mag, diff_frac, inlier_ratio, frob_norm = (
                    self._collect_pair_stats(prev_gray, curr_gray)
                )
            except Exception:
                mean_mag, diff_frac, inlier_ratio, frob_norm = (
                    0.0, 0.0, 0.0, None
                )
            is_dead = self._is_dead_from_stats(
                mean_mag, diff_frac, inlier_ratio, frob_norm
            )
            stats_list.append(
                {
                    "frame_index": pair_idx + 1,
                    "mean_mag": mean_mag,
                    "diff_fraction": diff_frac,
                    "inlier_ratio": inlier_ratio,
                    "frob_norm": frob_norm,
                    "is_dead": is_dead,
                }
            )
            decisions.append(not is_dead)
            prev_gray = curr_gray
            pair_idx += 1
            if pair_idx % 10 == 0 and total_frames > 1:
                pct = min(50, int(pair_idx * 50 / (total_frames - 1)))
                _emit(pct)
                if preview_cb:
                    preview_cb(curr_frame, pct)
        cap.release()
        _emit(50, "Detected")

        if auto and len(stats_list) > 0:
            sample = stats_list
            if auto_sample_limit > 0:
                sample = stats_list[:auto_sample_limit]
            self._compute_thresholds_from_stats(
                sample,
                lock_flow=(self.flow_threshold < 0.1),
                lock_area=(self.motion_area_fraction < 0.01),
            )
            _emit(55, "Calibrating")
            for i, s in enumerate(stats_list):
                old = s["is_dead"]
                new = self._is_dead_from_stats(
                    float(s["mean_mag"]),
                    float(s["diff_fraction"]),
                    float(s["inlier_ratio"]),
                    s["frob_norm"],
                )
                s["is_dead"] = new
                if i % 100 == 0:
                    _emit(55 + int(i / len(stats_list) * 15), "Reclassifying")
            decisions = [True]
            for s in stats_list:
                decisions.append(not bool(s["is_dead"]))
        return decisions


def run_deadframes(
    input_path: str,
    output_path: str,
    flow_threshold: float = 0.5,
    motion_area_fraction: float = 0.15,
    homography_inlier_ratio: float = 0.5,
    orb_features: int = 500,
    min_matches: int = 8,
    ransac_threshold: float = 5.0,
    detect_scale: float = 1.0,
    keep_talking: bool = False,
    keep_camera: bool = False,
    parallax_mode: bool = False,
    auto: bool = False,
    auto_sample_limit: int = 0,
    cadence: int = 3,
    small_movements: Optional[float] = None,
    prores: bool = False,
    no_audio: bool = False,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    preview_cb: Optional[Callable[[object, int], None]] = None,
) -> dict[str, Any]:
    if not DEADFRAMES_AVAILABLE:
        raise ImportError(
            "OpenCV (cv2) is required. Install with: pip install amverge[deadframes]"
        )

    if detect_scale >= 1.0:
        info = _get_video_info(input_path)
        detect_scale = min(1.0, 360.0 / max(1, info["height"]))

    detector = DeadFrameDetector(
        flow_threshold=flow_threshold,
        motion_area_fraction=motion_area_fraction,
        homography_inlier_ratio=homography_inlier_ratio,
        orb_features=orb_features,
        min_matches=min_matches,
        ransac_threshold=ransac_threshold,
        detect_scale=detect_scale,
        keep_talking=keep_talking,
        keep_camera=keep_camera,
        parallax_mode=parallax_mode,
    )

    if small_movements is not None:
        detector.flow_threshold = small_movements

    decisions = detector.process_video(
        input_path,
        auto=auto,
        auto_sample_limit=auto_sample_limit,
        progress_cb=progress_cb,
        preview_cb=preview_cb,
    )

    if progress_cb:
        progress_cb(70, "Smoothing cadence")

    decisions = _smooth_decisions(decisions, cadence)

    info = _get_video_info(input_path)
    fps = info["fps"]
    pix_fmt = info["pix_fmt"]

    segments = _decisions_to_segments(decisions, fps)
    total = len(decisions)
    kept = sum(1 for d in decisions if d)
    dropped = total - kept

    if dropped == 0 or kept == 0:
        import shutil
        if progress_cb:
            progress_cb(90, "Copying input")
        shutil.copy2(input_path, output_path)
        return {
            "total_frames": total,
            "kept_frames": kept,
            "dropped_frames": dropped,
            "segments": segments,
            "duration_after": sum(e - s for s, e in segments) if segments else 0,
            "flow_threshold": detector.flow_threshold,
            "motion_area_fraction": detector.motion_area_fraction,
        }

    has_audio = False
    audio_filter = ""
    if not no_audio:
        audio_info = _get_audio_info(input_path)
        if audio_info is not None:
            has_audio = True

    if has_audio:
        audio_filter = _build_audio_filter(segments)

    if progress_cb:
        progress_cb(75, "Building FFmpeg command")

    cmd = _build_ffmpeg_cmd(
        input_path=input_path,
        output_path=output_path,
        segments=segments,
        fps=fps,
        pix_fmt=pix_fmt,
        prores=prores,
        no_audio=no_audio,
        audio_filter=audio_filter,
    )

    if progress_cb:
        progress_cb(80, "Encoding")

    result = subprocess.run(
        cmd, capture_output=True, text=True,
        creationflags=_CREATE_NO_WINDOW,
    )
    if result.returncode != 0:
        tail = "\n".join(result.stderr.strip().splitlines()[-10:])
        raise RuntimeError(f"FFmpeg failed:\n{tail}")

    if progress_cb:
        progress_cb(100, "Done")

    return {
        "total_frames": total,
        "kept_frames": kept,
        "dropped_frames": dropped,
        "segments": segments,
        "duration_after": sum(e - s for s, e in segments),
        "flow_threshold": detector.flow_threshold,
        "motion_area_fraction": detector.motion_area_fraction,
    }
