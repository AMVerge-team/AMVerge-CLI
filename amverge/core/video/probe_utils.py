from __future__ import annotations

"""ffprobe wrappers for video metadata.

Convenience functions that shell out to ffprobe for common probe queries.
All functions accept a ``str`` or ``Path`` and return typed values.

Example:
    >>> from amverge.core.video.probe_utils import probe_video_fps, probe_video_dimensions
    >>> fps = probe_video_fps("episode.mp4")
    >>> w, h = probe_video_dimensions("episode.mp4")
"""

import json
import subprocess
import sys
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import av

from ..infra.binaries import get_ffprobe

# Without this every ffprobe call flashes its own console window when the CLI
# runs inside the windowed (--noconsole) app sidecar.
CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def probe_video_fps(input_video: str | Path) -> float:
    """Get the frame rate of the first video stream.

    Returns:
        FPS as a float (e.g. 23.976, 24.0, 30.0).
    """
    cmd = [
        get_ffprobe(),
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(input_video),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, check=True, creationflags=CREATE_NO_WINDOW
    )
    raw = result.stdout.strip().splitlines()[0].strip()
    if "/" in raw:
        num, den = map(int, raw.split("/"))
        return num / den if den != 0 else 0.0
    return float(raw)


def probe_video_dimensions(input_video: str | Path) -> tuple[int, int]:
    """Get the width and height of the first video stream.

    Returns:
        ``(width, height)`` tuple in pixels.
    """
    cmd = [
        get_ffprobe(),
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        str(input_video),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, check=True, creationflags=CREATE_NO_WINDOW
    )
    raw = result.stdout.strip().splitlines()[0].strip()
    width, height = map(int, raw.split("x"))
    return width, height


def probe_video_duration(input_video: str | Path) -> float:
    """Get the total duration of the video.

    Returns:
        Duration in seconds as a float.
    """
    cmd = [
        get_ffprobe(),
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(input_video),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, check=True, creationflags=CREATE_NO_WINDOW
    )
    raw = result.stdout.strip().splitlines()[0].strip()
    return float(raw)


def probe_video_total_frames(input_video: str | Path, video_fps: float, video_duration: float) -> int:
    """Estimate total frame count from FPS and duration.

    Args:
        video_fps: Frame rate from :func:`probe_video_fps`.
        video_duration: Duration from :func:`probe_video_duration`.

    Returns:
        Estimated frame count as ``int(fps * duration)``.
    """
    return int(video_fps * video_duration)


@lru_cache(maxsize=64)
def probe_seek_origin(input_video: str | Path) -> Fraction:
    """Seconds from the file's start time to its first video timestamp.

    ffmpeg's input ``-ss`` counts from the file's start time (the earliest
    stream start), but cut times count from the video's. They differ when
    another stream starts first: an MKV keeps AAC priming as a negative audio
    start (``-0.021``) where an MP4 hides it in an edit list, so an unshifted
    seek lands 21 ms, one frame, early. Add this to every input ``-ss``.

    Asks ffprobe, since it computes start times as ffmpeg will: PyAV reports
    0 for that MKV audio, and packet pts can't tell MKV priming (counted)
    from MP4 priming (hidden by the edit list).
    """
    cmd = [
        get_ffprobe(), "-v", "error", "-select_streams", "v:0",
        "-show_entries", "format=start_time:stream=start_time", "-of", "json", str(input_video),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True, creationflags=CREATE_NO_WINDOW
        )
        data = json.loads(result.stdout)
        video = Fraction(data["streams"][0]["start_time"])
        first = Fraction(data["format"]["start_time"])
    except Exception:
        return Fraction(0)
    return video - first


@lru_cache(maxsize=64)
def probe_video_rate(input_video: str | Path) -> Fraction | None:
    """Average frame rate of the first video stream as an exact fraction
    (``30000/1001``, not ``29.97``), or None if it cannot be read. Frame-exact
    cutting needs the exact value: a float rate drifts off the frame grid.
    """
    try:
        with av.open(str(input_video)) as container:
            rate = container.streams.video[0].average_rate
        return Fraction(rate) if rate else None
    except Exception:
        return None
