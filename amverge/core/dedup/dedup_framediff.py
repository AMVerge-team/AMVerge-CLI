from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from ._encode import build_stats, encode_selected, probe_frame_count

FRAMEDIFF_AVAILABLE = False
try:
    import cv2
    import numpy as np
    FRAMEDIFF_AVAILABLE = True
except ImportError:
    pass

_ANALYZE_MAX_WIDTH = 640


def dedup_framediff(
    video_path: str,
    output_path: str,
    threshold: float = 10.0,
    min_change_pct: float = 2.0,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> Tuple[str, dict]:
    """Remove near-duplicate frames by pixel-difference motion detection.

    A frame is kept when the fraction of pixels differing from the last kept
    frame (by more than ``threshold`` intensity) exceeds an adaptive area
    threshold: ``max(min_change_pct, median_noise_floor * 1.5)``. Analysis runs
    on a grayscale downscale; output is encoded from the full-resolution source
    with audio, color and bit depth preserved.

    Returns:
        (output_path, stats) with frames_in/out/removed/pct_removed.
    """
    if not FRAMEDIFF_AVAILABLE:
        raise ImportError(
            "FrameDiff dedup requires opencv. Run: pip install amverge[dedup]"
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

    if progress_cb:
        progress_cb(0, "Sampling noise floor...")

    prev_gray = _prep(frame)
    area_pixels = prev_gray.size
    sample_count = min(30, total_frames - 1)
    noise_fracs: List[float] = []
    for _ in range(sample_count):
        success, frame = cap.read()
        if not success:
            break
        curr_gray = _prep(frame)
        diff = cv2.absdiff(prev_gray, curr_gray)
        changed = int(np.count_nonzero(diff > threshold))
        noise_fracs.append(100.0 * changed / area_pixels)
        prev_gray = curr_gray

    if noise_fracs:
        ordered = sorted(noise_fracs)
        noise_floor = ordered[len(ordered) // 2]
    else:
        noise_floor = 0.0
    area_threshold_pct = max(min_change_pct, noise_floor * 1.5)

    cap.release()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to reopen: {video_path}")

    success, frame = cap.read()
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
        diff = cv2.absdiff(prev_gray, curr_gray)
        changed = int(np.count_nonzero(diff > threshold))
        changed_pct = 100.0 * changed / area_pixels

        if changed_pct > area_threshold_pct:
            keep_indices.append(frame_idx)
            prev_gray = curr_gray

        if progress_cb:
            pct = min(94, int((frame_idx / max(1, total_frames - 1)) * 95))
            if pct != last_pct:
                progress_cb(pct, f"Dedup (framediff)... {len(keep_indices)}/{frame_idx + 1}")
                last_pct = pct

    cap.release()
    frames_in = frame_idx + 1

    probe_n = probe_frame_count(video_path)
    if probe_n > 0 and abs(probe_n - frames_in) > max(2, int(0.01 * probe_n)):
        raise RuntimeError(
            f"Frame count mismatch (decoded {frames_in}, container {probe_n}) - "
            "source is likely VFR. Use the ffmpeg method or re-encode to CFR first."
        )

    if progress_cb:
        progress_cb(95, "Encoding output...")

    encode_selected(video_path, output_path, keep_indices,
                    progress_cb=progress_cb, progress_lo=95, progress_hi=99)
    stats = build_stats(frames_in, len(keep_indices))

    if progress_cb:
        progress_cb(100, f"Complete ({len(keep_indices)}/{frames_in} frames kept)")

    return output_path, stats
