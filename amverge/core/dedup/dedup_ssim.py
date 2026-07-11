from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from ..video.probe_utils import probe_video_fps
from ._encode import build_stats, encode_selected, probe_frame_count

SSIM_AVAILABLE = False
try:
    import cv2
    import numpy as np
    SSIM_AVAILABLE = True
except ImportError:
    pass

_ANALYZE_MAX_WIDTH = 640


if SSIM_AVAILABLE:
    def _windowed_ssim(img1: "np.ndarray", img2: "np.ndarray") -> float:
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        a = img1.astype(np.float64)
        b = img2.astype(np.float64)
        mu1 = cv2.GaussianBlur(a, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(b, (11, 11), 1.5)
        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu1_mu2
        ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
            (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
        )
        return float(ssim_map.mean())


def analyze_ssim(
    video_path: str,
    threshold: float = 0.987,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    progress_hi: int = 95,
) -> Tuple[List[int], int, float]:
    """Return (keep_indices, frames_in, fps). Keeps a frame when its windowed
    SSIM against the last kept frame is below threshold. Aborts on VFR sources
    whose decoded count diverges from the container count (indices misalign)."""
    if not SSIM_AVAILABLE:
        raise ImportError(
            "SSIM dedup requires opencv. Run: pip install amverge[dedup]"
        )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open: {video_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    scale = min(1.0, _ANALYZE_MAX_WIDTH / max(1, w))

    def _prep(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if scale < 1.0:
            gray = cv2.resize(gray, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_AREA)
        return gray

    success, frame = cap.read()
    if not success:
        cap.release()
        raise RuntimeError("No frames in video")

    keep_indices: List[int] = [0]
    prev_gray = _prep(frame)
    frame_idx = 0
    last_pct = -1

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_idx += 1

        curr_gray = _prep(frame)
        if _windowed_ssim(prev_gray, curr_gray) < threshold:
            keep_indices.append(frame_idx)
            prev_gray = curr_gray

        if progress_cb:
            pct = min(progress_hi - 1, int((frame_idx / max(1, total_frames - 1)) * progress_hi))
            if pct != last_pct:
                progress_cb(pct, f"Dedup (SSIM)... {len(keep_indices)}/{frame_idx + 1}")
                last_pct = pct

    cap.release()
    frames_in = frame_idx + 1

    probe_n = probe_frame_count(video_path)
    if probe_n > 0 and abs(probe_n - frames_in) > max(2, int(0.01 * probe_n)):
        raise RuntimeError(
            f"Frame count mismatch (decoded {frames_in}, container {probe_n}) - "
            "source is likely VFR. Use the ffmpeg method or re-encode to CFR first."
        )

    return keep_indices, frames_in, probe_video_fps(video_path)


def dedup_ssim(
    video_path: str,
    output_path: str,
    threshold: float = 0.987,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    codec: Optional[str] = None,
    crf: int = 18,
) -> Tuple[str, dict]:
    """Remove near-duplicate frames using structural similarity (SSIM).

    Analysis runs on a grayscale downscale; the output is encoded from the
    full-resolution source and preserves audio, color and bit depth.

    Returns:
        (output_path, stats) with frames_in/out/removed/pct_removed.
    """
    keep_indices, frames_in, _ = analyze_ssim(video_path, threshold, progress_cb)

    if progress_cb:
        progress_cb(95, "Encoding output...")

    encode_selected(video_path, output_path, keep_indices, crf=crf, codec=codec,
                    progress_cb=progress_cb, progress_lo=95, progress_hi=99)
    stats = build_stats(frames_in, len(keep_indices))

    if progress_cb:
        progress_cb(100, f"Complete ({len(keep_indices)}/{frames_in} frames kept)")

    return output_path, stats
