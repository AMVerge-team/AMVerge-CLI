"""Keyframe-based cut detection (primary method).

Cuts the video at I-frame boundaries extracted via PyAV.
Fast and lossless - no re-encoding needed.
"""
from __future__ import annotations

from typing import Callable

from ..keyframes import generate_keyframes
from ..video import merge_short_scenes

ProgressCb = Callable[[int, str], None]


def detect_cuts_by_keyframe(
    video_path: str,
    min_duration: float = 0.25,
    progress_cb: ProgressCb | None = None,
) -> list[float]:
    """Return cut-point timestamps (seconds) using keyframe packet metadata.

    Args:
        video_path: Path to the source video.
        min_duration: Merge any adjacent cuts closer than this many seconds.
        progress_cb: Optional ``(percent, message)`` callback.

    Returns:
        Sorted list of cut-point timestamps, not including 0.0.
    """
    keyframes = generate_keyframes(
        video_path,
        progress_cb=progress_cb,
        progress_base=0,
        progress_range=100,
    )

    if not keyframes:
        return []

    cut_points = sorted(keyframes[1:])
    cut_points = merge_short_scenes([0.0] + cut_points, min_duration=min_duration)[1:]

    return cut_points
