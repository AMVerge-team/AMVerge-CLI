"""Edge-detection scene scanner (alternative method).

Uses Canny edge maps + average-pooled cosine similarity to find hard cuts
inside keyframe windows. More frame-accurate than keyframe-only detection
but significantly slower and requires ``opencv-python-headless``.

Install the optional dep:
    pip install amverge-cli[edge]
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import numpy as np

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

import av

from ..keyframes import generate_keyframes
from ..video import merge_short_scenes

ProgressCb = Callable[[int, str], None]


def _magnitude(vec: np.ndarray) -> float:
    return math.sqrt(float(np.sum(vec ** 2)))


def _pool(arr: np.ndarray, dim: int) -> np.ndarray:
    h = (arr.shape[0] // dim) * dim
    w = (arr.shape[1] // dim) * dim
    arr = arr[:h, :w]
    return arr.reshape(h // dim, dim, w // dim, dim).mean(axis=(1, 3))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    af = a.flatten().astype(np.float32)
    bf = b.flatten().astype(np.float32)
    denom = _magnitude(af) * _magnitude(bf)
    return float(np.dot(af, bf) / denom) if denom != 0 else 1.0


def _scan_window(
    video_path: str,
    start_sec: float,
    end_sec: float,
    threshold: float,
    blocksize: int,
) -> list[float]:
    """Scan a time window for hard cuts via edge cosine dissimilarity."""
    cuts: list[float] = []

    with av.open(video_path) as container:
        stream = container.streams.video[0]
        container.seek(int(start_sec * 1_000_000), any_frame=False, backward=True)

        prev: np.ndarray | None = None

        for frame in container.decode(stream):
            if frame.time is None:
                continue
            ts = float(frame.time)
            if ts < start_sec:
                continue
            if ts > end_sec:
                break

            img = frame.reformat(width=480, height=270, format="gray").to_ndarray()
            edges = cv2.Canny(img, 50, 100)

            if np.count_nonzero(edges) == 0:
                continue

            pooled = _pool(edges, blocksize)

            if prev is not None and abs(1.0 - _cosine(pooled, prev)) > threshold:
                cuts.append(ts)

            prev = pooled

    return cuts


def _keyframe_windows(keyframes: list[float], radius: float) -> list[tuple[float, float]]:
    return [(max(0.0, kf - radius), kf + radius) for kf in keyframes]


def detect_cuts_by_edge(
    video_path: str,
    threshold: float = 0.15,
    radius: float = 0.6,
    blocksize: int = 3,
    min_duration: float = 0.5,
    workers: int = 4,
    progress_cb: ProgressCb | None = None,
) -> list[float]:
    """Return cut-point timestamps (seconds) using edge-map cosine dissimilarity.

    Scans keyframe windows rather than the full video to keep runtime manageable.
    Requires ``opencv-python-headless`` (``pip install amverge-cli[edge]``).

    Args:
        video_path: Path to the source video.
        threshold: Cosine dissimilarity threshold for a cut (higher = more cuts).
        radius: Seconds around each keyframe to scan.
        blocksize: Pooling block size for edge maps (larger = blurrier comparison).
        min_duration: Merge adjacent cuts closer than this many seconds.
        workers: Parallel scanning threads.
        progress_cb: Optional ``(percent, message)`` callback.

    Returns:
        Sorted list of cut-point timestamps, not including 0.0.
    """
    if not _CV2_AVAILABLE:
        raise ImportError(
            "Edge detection requires opencv-python-headless. "
            "Install with: pip install amverge-cli[edge]"
        )

    def _cb(pct: int, msg: str) -> None:
        if progress_cb:
            try:
                progress_cb(pct, msg)
            except Exception:
                pass

    _cb(5, "Extracting keyframes for window building...")
    keyframes = generate_keyframes(video_path, progress_cb=None)

    if not keyframes:
        return []

    windows = _keyframe_windows(keyframes, radius)
    total = max(len(windows), 1)
    all_cuts: list[float] = []

    _cb(15, f"Scanning {total} keyframe windows...")

    with ThreadPoolExecutor(max_workers=min(workers, os.cpu_count() or 4)) as executor:
        futures = {
            executor.submit(_scan_window, video_path, s, e, threshold, blocksize): i
            for i, (s, e) in enumerate(windows)
        }
        done = 0
        for future in as_completed(futures):
            try:
                all_cuts.extend(future.result())
            except Exception:
                pass
            done += 1
            pct = 15 + int(80 * done / total)
            _cb(pct, f"Scanned {done}/{total} windows")

    cut_points = sorted(set(round(t, 6) for t in all_cuts))
    if cut_points:
        boundaries = [0.0] + cut_points
        cut_points = merge_short_scenes(boundaries, min_duration=min_duration)[1:]

    _cb(100, f"Edge detection done - {len(cut_points)} cuts")
    return cut_points
