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


@lru_cache(maxsize=64)
def _probe_basic_metadata(
    path_str: str, size: int, mtime_ns: int
) -> tuple[float | None, float | None, int | None, int | None, str | None, Fraction | None]:
    del size, mtime_ns
    try:
        with av.open(path_str) as container:
            stream = next(iter(container.streams.video), None)
            if stream is None:
                return None, None, None, None, None, None
            rate = stream.average_rate
            base_rate = stream.base_rate or rate
            fps = float(base_rate) if base_rate else None
            duration = (
                float(container.duration / av.time_base)
                if container.duration is not None
                else (
                    float(stream.duration * stream.time_base)
                    if stream.duration is not None and stream.time_base is not None
                    else None
                )
            )
            width = stream.codec_context.width or None
            height = stream.codec_context.height or None
            codec = stream.codec_context.name or None
            avg_rate = Fraction(rate) if rate else None
            return fps, duration, width, height, codec, avg_rate
    except Exception:
        return None, None, None, None, None, None


def _cached_meta(input_video: str | Path):
    p = Path(input_video)
    try:
        st = p.stat()
        return _probe_basic_metadata(str(p), st.st_size, st.st_mtime_ns)
    except OSError:
        return None, None, None, None, None, None


def probe_video_fps(input_video: str | Path) -> float:
    """Get the frame rate of the first video stream.

    Returns:
        FPS as a float (e.g. 23.976, 24.0, 30.0).
    """
    fps, _, _, _, _, _ = _cached_meta(input_video)
    if fps is not None and fps > 0:
        return fps
    cmd = [
        get_ffprobe(),
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(input_video),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True, creationflags=CREATE_NO_WINDOW
    )
    raw = result.stdout.strip().splitlines()[0].strip()
    if "/" in raw:
        num, den = map(int, raw.split("/"))
        return num / den if den != 0 else 0.0
    return float(raw)


def probe_video_codec(input_video: str | Path) -> str | None:
    """Codec name of the first video stream (``"h264"``, ``"mpeg4"``...), or None
    when the file has no readable video stream."""
    _, _, _, _, codec, _ = _cached_meta(input_video)
    return codec


def probe_video_dimensions(input_video: str | Path) -> tuple[int, int]:
    """Get the width and height of the first video stream.

    Returns:
        ``(width, height)`` tuple in pixels.

    Read as json by name: csv output appends an empty field per side-data block
    (a Dolby Vision stream prints ``3840x2160x``), which broke positional parsing.
    """
    _, _, width, height, _, _ = _cached_meta(input_video)
    if width and height:
        return int(width), int(height)
    cmd = [
        get_ffprobe(),
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        str(input_video),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True, creationflags=CREATE_NO_WINDOW
    )
    stream = json.loads(result.stdout)["streams"][0]
    return int(stream["width"]), int(stream["height"])


def probe_video_duration(input_video: str | Path) -> float:
    """Get the total duration of the video.

    Returns:
        Duration in seconds as a float.
    """
    _, duration, _, _, _, _ = _cached_meta(input_video)
    if duration is not None and duration > 0:
        return duration
    cmd = [
        get_ffprobe(),
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(input_video),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True, creationflags=CREATE_NO_WINDOW
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
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True, creationflags=CREATE_NO_WINDOW
        )
        data = json.loads(result.stdout)
        video = Fraction(data["streams"][0]["start_time"])
        first = Fraction(data["format"]["start_time"])
    except Exception:
        return Fraction(0)
    return video - first


@lru_cache(maxsize=64)
def probe_video_start(input_video: str | Path) -> float:
    """Raw timestamp of the first video frame, in seconds, or 0.0 when unknown.

    Cut times count from the video's start, but PyAV packet timestamps are
    raw: an MPEG-TS stream starts around 1.4 s, so its keyframes need this
    subtracted before they line up with ffmpeg's clock and the app's.
    """
    cmd = [
        get_ffprobe(), "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=start_time", "-of", "default=nk=1:nw=1", str(input_video),
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True, creationflags=CREATE_NO_WINDOW
        )
        return float(result.stdout.strip().splitlines()[0])
    except Exception:
        return 0.0


@lru_cache(maxsize=64)
def probe_video_rate(input_video: str | Path) -> Fraction | None:
    """Average frame rate of the first video stream as an exact fraction
    (``30000/1001``, not ``29.97``), or None if it cannot be read. Frame-exact
    cutting needs the exact value: a float rate drifts off the frame grid.
    """
    _, _, _, _, _, rate = _cached_meta(input_video)
    return rate
