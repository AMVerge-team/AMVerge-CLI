from __future__ import annotations

"""Scene array conversion utilities.

Converts between frame-based scene arrays (from TransNetV2) and
second-based scene arrays, and builds structured scene dict lists.

Example:
    >>> from amverge.core.video.scene_utils import scenes_to_objects
    >>> import numpy as np
    >>> scenes_secs = np.array([[0.0, 5.0], [5.0, 10.0]])
    >>> scenes_frames = np.array([[0, 120], [120, 240]])
    >>> objs = scenes_to_objects(scenes_secs, scenes_frames)
    >>> objs[0]["scene_index"]
    0
"""

import numpy as np

from .probe_utils import probe_video_fps


def scenes_frames_to_seconds(scenes: np.ndarray, fps: float) -> np.ndarray:
    """Convert frame-based scene boundaries to seconds.

    Args:
        scenes: ``(N, 2)`` ndarray of ``[start_frame, end_frame]``.
        fps: Video frame rate.

    Returns:
        ``(N, 2)`` ndarray of ``[start_sec, end_sec]`` rounded to 2 decimals.
    """
    return np.round(scenes / fps, 2)


def transnet_scenes_to_seconds(scenes: np.ndarray, fps: float) -> np.ndarray:
    """Convert TransNetV2 frame ranges to contiguous ``[start_sec, end_sec)``.

    ``predictions_to_scenes`` returns *inclusive* end frames (a scene covering
    frames 0-89 is ``[0, 89]``), so the exclusive end is the frame after it:
    ``end_sec`` is the next scene's ``start_sec``, and the last scene ends at
    the video's duration. Kept to microsecond precision; rounding to
    centiseconds (what :func:`scenes_frames_to_seconds` does) moves the end
    boundary by up to 5 ms, enough to drop or add a frame at 60 fps.

    Args:
        scenes: ``(N, 2)`` ndarray of ``[start_frame, end_frame_inclusive]``.
        fps: Video frame rate.

    Returns:
        ``(N, 2)`` float ndarray of ``[start_sec, end_sec]``.
    """
    frames = np.asarray(scenes, dtype=np.float64).reshape(-1, 2)
    exclusive = frames + np.array([0.0, 1.0])
    return np.round(exclusive / fps, 6)


def convert_scenes_to_timestamps(
    src_video: str, scenes: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Extract cut point timestamps from a scene array.

    Cut points are the end frames of all but the last scene.

    Args:
        src_video: Path to video (for FPS probe).
        scenes: ``(N, 2)`` ndarray of ``[start_frame, end_frame]``.

    Returns:
        Tuple of ``(timestamps_sec, cut_frames)``.
    """
    fps = probe_video_fps(src_video)
    cuts = scenes[:-1, 1]
    timestamps = cuts / fps
    return timestamps, cuts


def scenes_to_objects(
    scenes_secs: np.ndarray, scenes_frames: np.ndarray
) -> list[dict]:
    """Build a list of scene dicts from second and frame arrays.

    Args:
        scenes_secs: ``(N, 2)`` ndarray of ``[start_sec, end_sec]``.
        scenes_frames: ``(N, 2)`` ndarray of ``[start_frame, end_frame]``.

    Returns:
        List of dicts with keys: ``scene_index``, ``start_sec``,
        ``end_sec``, ``duration_sec``, ``start_frame``, ``end_frame``.
    """
    scenes: list[dict] = []
    total = min(len(scenes_secs), len(scenes_frames))
    for scene_index in range(total):
        sec_pair = scenes_secs[scene_index]
        frame_pair = scenes_frames[scene_index]
        scenes.append(
            {
                "scene_index": scene_index,
                "start_sec": float(sec_pair[0]),
                "end_sec": float(sec_pair[1]),
                "duration_sec": max(0.0, float(sec_pair[1]) - float(sec_pair[0])),
                "start_frame": int(frame_pair[0]),
                "end_frame": int(frame_pair[1]),
            }
        )
    return scenes
