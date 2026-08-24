"""Fold ultra-short scenes into whichever neighbour they resemble.

A scene of a few frames cannot be stream-copied accurately: the copy starts at
the preceding keyframe and ffmpeg marks the pre-start packets discardable, so the
clip ends up holding a whole GOP under a duration meant for a fraction of it.
Those clips play far too fast and their poster frame never decodes.

Rather than special-casing them at cut time, they are merged away here: each one
is compared against the scene before and after it, and joined to the closer
match, so the cut list only ever contains ranges long enough to cut cleanly.
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np

from ..infra.binaries import get_ffmpeg

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

# Below this a scene is a handful of frames and not worth its own clip.
MIN_SCENE_SEC = 0.25

# Comparison frames are tiny: this is a "which of these two is closer" test, and
# a thumbnail-sized grab would cost far more than the answer is worth.
_SAMPLE_W = 32
_SAMPLE_H = 18
_SAMPLE_BYTES = _SAMPLE_W * _SAMPLE_H * 3


def _grab_frame(video_path: str, at_sec: float) -> np.ndarray | None:
    """One downscaled RGB frame, or None if it could not be read."""
    try:
        result = subprocess.run(
            [
                get_ffmpeg(), "-v", "error",
                "-ss", f"{max(0.0, at_sec):.3f}",
                "-i", video_path,
                "-frames:v", "1",
                "-vf", f"scale={_SAMPLE_W}:{_SAMPLE_H}",
                "-pix_fmt", "rgb24",
                "-f", "rawvideo", "pipe:1",
            ],
            capture_output=True, timeout=20, creationflags=CREATE_NO_WINDOW,
        )
    except Exception:
        return None

    if len(result.stdout) < _SAMPLE_BYTES:
        return None
    return np.frombuffer(result.stdout[:_SAMPLE_BYTES], dtype=np.uint8).astype(np.float32)


def _similarity(a: np.ndarray | None, b: np.ndarray | None) -> float:
    """Cosine similarity of two frames; -1 when either is missing."""
    if a is None or b is None:
        return -1.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else -1.0


def _midpoint(scene: dict) -> float:
    return (float(scene["start_sec"]) + float(scene["end_sec"])) / 2.0


def merge_short_scenes(
    scenes: list[dict],
    video_path: str,
    min_sec: float = MIN_SCENE_SEC,
) -> list[dict]:
    """Merge scenes shorter than ``min_sec`` into their closest-looking neighbour.

    Scenes are contiguous, so a merge just extends the neighbour's range over the
    short scene. Indexes are renumbered afterwards.

    Args:
        scenes: Scene dicts from :func:`scenes_to_objects`.
        video_path: Source video, sampled for the comparison frames.
        min_sec: Scenes shorter than this are merged away.

    Returns:
        A new scene list. Returned unchanged when nothing is short enough.
    """
    if len(scenes) < 2:
        return scenes

    if all(float(s["end_sec"]) - float(s["start_sec"]) >= min_sec for s in scenes):
        return scenes

    merged: list[dict] = []
    for scene in scenes:
        duration = float(scene["end_sec"]) - float(scene["start_sec"])
        has_next = scene is not scenes[-1]

        if duration >= min_sec or (not merged and not has_next):
            merged.append(dict(scene))
            continue

        index = scenes.index(scene)
        prev_scene = merged[-1] if merged else None
        next_scene = scenes[index + 1] if index + 1 < len(scenes) else None

        if prev_scene is None:
            _extend_into_next(scene, next_scene)
            continue
        if next_scene is None:
            _extend_into_prev(prev_scene, scene)
            continue

        sample = _grab_frame(video_path, _midpoint(scene))
        to_prev = _similarity(sample, _grab_frame(video_path, _midpoint(prev_scene)))
        to_next = _similarity(sample, _grab_frame(video_path, _midpoint(next_scene)))

        if to_prev >= to_next:
            _extend_into_prev(prev_scene, scene)
        else:
            _extend_into_next(scene, next_scene)

    for position, scene in enumerate(merged):
        scene["scene_index"] = position
        scene["duration_sec"] = float(scene["end_sec"]) - float(scene["start_sec"])

    return merged


def _extend_into_prev(prev_scene: dict, short: dict) -> None:
    """Absorb ``short`` into the scene before it."""
    prev_scene["end_sec"] = short["end_sec"]
    if "end_frame" in short:
        prev_scene["end_frame"] = short["end_frame"]


def _extend_into_next(short: dict, next_scene: dict | None) -> None:
    """Absorb ``short`` into the scene after it, which is not emitted yet."""
    if next_scene is None:
        return
    next_scene["start_sec"] = short["start_sec"]
    if "start_frame" in short:
        next_scene["start_frame"] = short["start_frame"]
