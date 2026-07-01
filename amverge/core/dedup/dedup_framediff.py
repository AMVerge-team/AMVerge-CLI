from __future__ import annotations

import os
import subprocess
import sys
from typing import Callable, Optional, Dict, Tuple

from ..infra.binaries import get_ffmpeg, get_ffprobe

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

FRAMEDIFF_AVAILABLE = False
try:
    import cv2
    import numpy as np
    FRAMEDIFF_AVAILABLE = True
except ImportError:
    pass


def dedup_framediff(
    video_path: str,
    output_path: str,
    threshold: float = 10.0,
    min_change_pct: float = 2.0,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> Tuple[str, Dict]:
    """Remove duplicate frames by pixel difference comparison.

    Compares consecutive grayscale frames via absdiff. A frame is kept
    only if more than min_change_pct% of pixels differ by > threshold.

    Args:
        video_path: Path to input video.
        output_path: Path for output video.
        threshold: Pixel intensity difference threshold (0-255).
        min_change_pct: Minimum percentage of changed pixels to keep a frame.
        progress_cb: Optional (pct, msg) callback.

    Returns:
        Output path on success.
    """
    if not FRAMEDIFF_AVAILABLE:
        raise ImportError(
            "FrameDiff dedup requires opencv. "
            "Run: pip install opencv-python"
        )

    ffmpeg = get_ffmpeg()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open: {video_path}")

    fps_val = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    total_pixels = w * h

    tmp_path = output_path + ".tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(tmp_path, fourcc, fps_val, (w, h))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create output: {tmp_path}")

    success, prev_frame = cap.read()
    if not success:
        cap.release()
        out.release()
        raise RuntimeError("No frames in video")

    if progress_cb:
        progress_cb(0, "Sampling adaptive threshold...")

    sample_count = min(30, total_frames - 1)
    total_movement = 0.0
    sample_frames = 0
    while sample_frames < sample_count:
        success, curr_frame = cap.read()
        if not success:
            break
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        total_movement += float(np.sum(cv2.absdiff(prev_gray, curr_gray)))
        prev_frame = curr_frame
        sample_frames += 1

    average_diff = total_movement / max(1, sample_frames) / max(1, total_pixels)
    adaptive_threshold = threshold + average_diff

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success, prev_frame = cap.read()

    out.write(prev_frame)
    frame_idx = 0
    saved = 1
    last_pct = -1

    while True:
        success, curr_frame = cap.read()
        if not success:
            break
        frame_idx += 1

        if progress_cb:
            pct = min(99, int((frame_idx / max(1, total_frames - 1)) * 100))
            if pct != last_pct:
                progress_cb(pct, f"Dedup (framediff)... {saved}/{frame_idx + 1}")
                last_pct = pct

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        frame_diff = cv2.absdiff(prev_gray, curr_gray)
        changed_pixels = np.count_nonzero(frame_diff > adaptive_threshold)

        if changed_pixels > total_pixels * (min_change_pct / 100.0):
            out.write(curr_frame)
            saved += 1
            prev_frame = curr_frame

    cap.release()
    out.release()

    if progress_cb:
        progress_cb(95, "Encoding output...")

    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", tmp_path,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600,
                       creationflags=CREATE_NO_WINDOW)
    if os.path.exists(tmp_path):
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    if r.returncode != 0:
        raise RuntimeError(f"FFmpeg encode failed: {r.stderr.strip()}")

    stats = {
        "frames_in": frame_idx + 1,
        "frames_out": saved,
        "frames_removed": max(0, (frame_idx + 1) - saved),
        "pct_removed": round((1 - saved / max(1, frame_idx + 1)) * 100, 1),
    }

    if progress_cb:
        progress_cb(100, f"Complete ({saved}/{frame_idx + 1} frames kept, {stats['pct_removed']}% removed)")

    return output_path, stats
