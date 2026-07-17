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


def detect_scenes_by_keyframe(
    video_path: str,
    min_duration: float = 0.25,
    progress_cb: ProgressCb | None = None,
) -> list[list[float]]:
    """Detect scenes by cutting at keyframe (I-frame) boundaries.

    Returns a list of ``[start_sec, end_sec]`` pairs packing the whole video
    from ``0.0`` to its duration, one scene per keyframe interval. Because every
    boundary is an I-frame timestamp, each scene starts on a keyframe and can be
    cut losslessly (copy) downstream.

    Mirrors the second-boundary shape produced by the TransNetV2 detector, so
    the cutting/manifest logic stays detection-agnostic. Container-agnostic via
    PyAV (mp4/mkv/webm/...).

    Args:
        video_path: Path to the source video.
        min_duration: Merge adjacent keyframes closer than this many seconds.
        progress_cb: Optional ``(percent, message)`` callback.

    Returns:
        Non-empty list of ``[start_sec, end_sec]`` scene pairs.
    """
    from ..video.probe_utils import probe_video_duration

    duration = float(probe_video_duration(video_path) or 0.0)
    cut_points = detect_cuts_by_keyframe(
        video_path, min_duration=min_duration, progress_cb=progress_cb
    )

    bounds: list[float] = [0.0]
    for c in sorted(float(x) for x in cut_points):
        # Keep strictly increasing and strictly inside (0, duration) so the
        # final scene closes cleanly on the video's end rather than a near-dupe.
        if c > bounds[-1] + 1e-6 and (duration <= 0.0 or c < duration - 1e-6):
            bounds.append(c)

    # Close the last scene on the real duration; if duration is unknown, extend
    # past the last boundary so a valid final scene still exists.
    end = duration if duration > bounds[-1] + 1e-6 else bounds[-1] + max(min_duration, 0.04)
    bounds.append(end)

    return [[bounds[i], bounds[i + 1]] for i in range(len(bounds) - 1)]
