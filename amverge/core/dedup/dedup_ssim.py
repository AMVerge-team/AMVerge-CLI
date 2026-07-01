from __future__ import annotations

import os
import subprocess
import sys
from typing import Callable, Optional, Dict, Tuple

from ..infra.binaries import get_ffmpeg, get_ffprobe

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

SSIM_AVAILABLE = False
try:
    import cv2
    import numpy as np
    from skimage.metrics import structural_similarity as compare_ssim
    _CPU_SSIM_FN = compare_ssim
    SSIM_AVAILABLE = True
except ImportError:
    pass


if SSIM_AVAILABLE:
    def _compute_ssim_gpu(img1_gray: np.ndarray, img2_gray: np.ndarray) -> float:
        try:
            import cupy as cp
            img1_gpu = cp.asarray(img1_gray, dtype=cp.float32)
            img2_gpu = cp.asarray(img2_gray, dtype=cp.float32)
            C1 = (0.01 * 255) ** 2
            C2 = (0.03 * 255) ** 2
            mu1 = cp.mean(img1_gpu)
            mu2 = cp.mean(img2_gpu)
            sigma1_sq = cp.var(img1_gpu)
            sigma2_sq = cp.var(img2_gpu)
            sigma12 = cp.mean((img1_gpu - mu1) * (img2_gpu - mu2))
            numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
            denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
            return float(cp.asnumpy(numerator / denominator))
        except Exception:
            pass
        return float(_CPU_SSIM_FN(img1_gray, img2_gray))
else:
    def _compute_ssim_gpu(img1_gray, img2_gray):
        raise ImportError("SSIM dedup requires opencv and scikit-image")


def dedup_ssim(
    video_path: str,
    output_path: str,
    threshold: float = 0.987,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> Tuple[str, Dict]:
    if not SSIM_AVAILABLE:
        raise ImportError(
            "SSIM dedup requires opencv and scikit-image. "
            "Run: pip install opencv-python scikit-image"
        )

    ffmpeg = get_ffmpeg()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open: {video_path}")

    fps_val = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

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
                progress_cb(pct, f"Dedup (SSIM)... {saved}/{frame_idx + 1}")
                last_pct = pct

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        ssim_val = _compute_ssim_gpu(prev_gray, curr_gray)

        if ssim_val < threshold:
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
