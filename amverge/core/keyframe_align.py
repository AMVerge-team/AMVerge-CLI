from __future__ import annotations

"""Keyframe extraction and scene-to-keyframe alignment.

Extracts keyframe timestamps via PyAV packet demux (no frame decode) and
classifies scenes by whether their boundaries align with a keyframe. This
determines which scenes can use lossless copy vs. need re-encoding.

Usage:
    >>> from amverge.core.keyframe_align import (
    ...     get_keyframe_timestamps_pyav,
    ...     classify_scenes_by_keyframe_alignment,
    ... )
    >>> keyframes = get_keyframe_timestamps_pyav("episode.mp4")
    >>> scenes = [(0.0, 5.0), (5.2, 10.0)]
    >>> copy, reencode = classify_scenes_by_keyframe_alignment(scenes, keyframes)
    >>> print(f"{len(copy)} lossless, {len(reencode)} re-encode")
"""

from bisect import bisect_left, bisect_right

import av


def get_keyframe_timestamps_pyav(video_path: str) -> list[float]:
    """Extract keyframe timestamps using PyAV packet demux.

    Uses ``stream.discard = nonkey`` to skip non-keyframe packets at the
    demux level - no frame decoding occurs. Returns deduplicated, sorted
    timestamps in seconds rounded to 2 decimal places.

    Uses ``av.Discard.nonkey`` enum (PyAV 17.x) with fallback to older
    ``"NONKEY"`` string for backward compatibility.

    Args:
        video_path: Path to the source video file.

    Returns:
        Sorted list of unique keyframe timestamps in seconds.

    Example:
        >>> kf = get_keyframe_timestamps_pyav("episode.mp4")
        >>> print(f"{len(kf)} keyframes, first at {kf[0]}s")
    """
    keyframe_times: list[float] = []
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        try:
            stream.discard = type(stream.discard).nonkey
        except (AttributeError, KeyError):
            pass
        for packet in container.demux(stream):
            if not packet.is_keyframe:
                continue
            ts = packet.pts if packet.pts is not None else packet.dts
            if ts is None:
                continue
            keyframe_times.append(round(float(ts * packet.time_base), 2))
    return sorted(set(keyframe_times))


def _nearest_within_threshold(
    sorted_keyframes: list[float], ts: float, threshold: float
) -> tuple[float | None, float | None]:
    i = bisect_left(sorted_keyframes, ts)
    candidates = []
    if i < len(sorted_keyframes):
        candidates.append(sorted_keyframes[i])
    if i > 0:
        candidates.append(sorted_keyframes[i - 1])
    if not candidates:
        return None, None
    nearest = min(candidates, key=lambda k: abs(k - ts))
    diff = abs(nearest - ts)
    if diff <= threshold:
        return nearest, diff
    return None, diff


def classify_scenes_by_keyframe_alignment(
    scenes_secs: list[tuple[float, float]],
    keyframe_timestamps: list[float],
    threshold: float = 0.2,
) -> tuple[list[dict], list[dict]]:
    """Partition scenes into lossless-copy and re-encode candidates.

    A scene is a copy candidate if its start time aligns with a keyframe
    within ``threshold`` seconds (default 0.2s).

    Args:
        scenes_secs: List of ``(start, end)`` tuples in seconds.
        keyframe_timestamps: Sorted keyframe timestamps from
            :func:`get_keyframe_timestamps_pyav`.
        threshold: Max allowed gap between scene start and nearest keyframe
            for copy eligibility.

    Returns:
        Tuple of ``(copy_candidates, reencode_candidates)``. Each candidate
        dict contains:
        - ``scene_id``: int, index in input list
        - ``orig_start``: float, original scene start
        - ``orig_end``: float, original scene end
        - ``start``: float, snapped or original start
        - ``start_snapped``: bool, whether start was aligned
        - ``start_diff_sec``: float, gap to nearest keyframe
        - ``mode``: ``"copy_candidate"`` or ``"reencode_candidate"``

    Example:
        >>> scenes = [(0.0, 5.0), (5.2, 10.0), (10.0, 15.0)]
        >>> kf = [0.0, 5.0, 10.0, 15.0]
        >>> copy, reencode = classify_scenes_by_keyframe_alignment(scenes, kf)
        >>> len(copy)     # scenes 0 and 2 start on keyframes
        2
        >>> len(reencode) # scene 1 starts at 5.2s, no keyframe within 0.2s
        1
    """
    if threshold < 0:
        raise ValueError(f"Cannot have negative threshold ({threshold})")

    kf = sorted(float(x) for x in keyframe_timestamps)
    copy_candidates: list[dict] = []
    reencode_candidates: list[dict] = []

    for idx, scene in enumerate(scenes_secs):
        scene_start = float(scene[0])
        scene_end = float(scene[1])

        snapped_start, start_diff = _nearest_within_threshold(kf, scene_start, threshold)
        snapped_end, end_diff = _nearest_within_threshold(kf, scene_end, threshold)

        mode = "copy_candidate" if snapped_start is not None else "reencode_candidate"

        record = {
            "scene_id": idx,
            "orig_start": scene_start,
            "orig_end": scene_end,
            "start": snapped_start if snapped_start is not None else scene_start,
            "end": snapped_end if snapped_end is not None else scene_end,
            "start_snapped": snapped_start is not None,
            "end_snapped": snapped_end is not None,
            "start_diff_sec": start_diff,
            "end_diff_sec": end_diff,
            "mode": mode,
        }

        if mode == "copy_candidate":
            copy_candidates.append(record)
        else:
            reencode_candidates.append(record)

    return copy_candidates, reencode_candidates
