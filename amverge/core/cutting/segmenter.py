"""FFmpeg segment-based scene cutting."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from ..infra.binaries import get_ffmpeg
from ..infra.ipc import log
from ..video.probe_utils import probe_video_duration
from .smart_cut import _trim_trailing_partial_gop

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

_COPY_VIDEO_ARGS = ["-c:v", "copy"]
_REENCODE_VIDEO_ARGS = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "16", "-pix_fmt", "yuv420p"]

_SILENCED = [
    re.compile(r"track\s+\d+:\s+codec frame size is not set", re.IGNORECASE),
    re.compile(r"^\[segment\s+@\s+[^\]]+\]\s+Opening\s+'.+'\s+for writing$", re.IGNORECASE),
]

CHUNK_SIZE = 1500
SEGMENT_TIME_EPSILON = 0.001


def _segment_boundaries(cut_points: list[float], total_duration: float) -> list[float]:
    """Absolute ``[start_0, start_1, ..., start_N, total_duration]`` boundaries
    for the segments ``run_ffmpeg_segment*`` produces from ``cut_points`` --
    same scheme :func:`collect_scenes` uses to report scene start/end."""
    return [0.0, *cut_points, total_duration]


def _trim_segment(path: str, duration_sec: float) -> None:
    """Drop trailing frames a segment-muxer split dragged in from the next
    scene (same GOP-boundary issue :func:`smart_cut.cut_scene` guards
    against -- the segment muxer has no keyframe-alignment awareness at all,
    so every split here is exposed to it). Best-effort: a probe/remux hiccup
    leaves the untrimmed segment rather than failing the whole cut.
    """
    if duration_sec is None or duration_sec <= 0:
        return
    try:
        _trim_trailing_partial_gop(Path(path), duration_sec)
    except Exception:
        pass


def _fmt_ts(seconds: float) -> str:
    return f"{float(seconds):.6f}".rstrip("0").rstrip(".")


def _fmt_segment_times(cut_points: list[float]) -> str:
    return ",".join(_fmt_ts(max(0.0, p - SEGMENT_TIME_EPSILON)) for p in cut_points)


def _clean_ffmpeg_output(text: str | None) -> str:
    if not text:
        return ""
    lines = [l for l in text.splitlines() if l.strip() and not any(p.search(l.strip()) for p in _SILENCED)]
    return "\n".join(lines)


def _run_chunk(
    video_path: str,
    output_pattern: str,
    cut_points: list[float],
    start_num: int,
    start_time: float,
    end_time: float | None,
    ffmpeg: str,
    video_args: list[str],
) -> None:
    cmd = [ffmpeg, "-y"]

    if start_time > 0.0:
        cmd += ["-ss", _fmt_ts(start_time)]
    if end_time is not None:
        cmd += ["-to", _fmt_ts(end_time)]

    cmd += [
        "-i", video_path,
        "-map", "0:v:0",
        "-map", "0:a?",
        *video_args,
        "-c:a", "aac",
        "-b:a", "160k",
        "-ac", "2",
        "-ar", "48000",
        "-f", "segment",
        "-segment_times", _fmt_segment_times(cut_points),
        "-segment_start_number", str(start_num),
        "-reset_timestamps", "1",
        output_pattern,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, creationflags=CREATE_NO_WINDOW)

    if result.returncode != 0:
        tail = result.stderr[-2000:] if result.stderr else "no stderr"
        raise RuntimeError(f"ffmpeg exit {result.returncode}: {tail}")


def run_ffmpeg_segment(
    video_path: str,
    output_pattern: str,
    cut_points: list[float],
    ffmpeg: str | None = None,
    total_duration: float | None = None,
) -> None:
    """Cut a video at specified timestamps using FFmpeg segment muxer.

    Uses stream copy (no re-encode) with AAC audio, falling back to a full
    re-encode if the copy fails (a codec the container has no tag for, e.g.
    ProRes or HuffYUV stream-copied into MP4). Chunks into 1500-cut batches
    to stay under the Windows 32,767-char command line limit.

    The segment muxer has no keyframe-alignment awareness -- any split that
    doesn't land on a keyframe can drag a few trailing frames from the next
    scene into the current one (same GOP-boundary issue smart_cut.cut_scene
    guards against). Each produced segment is trimmed back to its intended
    duration afterward to catch that.

    Args:
        video_path: Path to the source video.
        output_pattern: FFmpeg output pattern (e.g. ``"out_%04d.mp4"``).
        cut_points: Sorted list of cut timestamps in seconds.
        ffmpeg: Optional path to ffmpeg binary. Auto-detected if None.
        total_duration: Video duration in seconds, for the last segment's
            end boundary. Probed from ``video_path`` if not given.
    """
    ff = ffmpeg or get_ffmpeg()

    def _run_all(video_args: list[str]) -> None:
        if len(cut_points) <= CHUNK_SIZE:
            _run_chunk(video_path, output_pattern, cut_points, 0, 0.0, None, ff, video_args)
        else:
            for i in range(0, len(cut_points), CHUNK_SIZE):
                chunk = cut_points[i: i + CHUNK_SIZE]
                start_time = cut_points[i - 1] if i > 0 else 0.0
                end_time = chunk[-1] if i + CHUNK_SIZE < len(cut_points) else None
                relative = [p - start_time for p in chunk]
                _run_chunk(video_path, output_pattern, relative, i, start_time, end_time, ff, video_args)

    try:
        _run_all(_COPY_VIDEO_ARGS)
    except RuntimeError as exc:
        log(f"Stream copy failed ({exc}), re-encoding instead")
        _run_all(_REENCODE_VIDEO_ARGS)

    duration = total_duration if total_duration is not None else probe_video_duration(video_path)
    boundaries = _segment_boundaries(cut_points, duration)
    for idx in range(len(boundaries) - 1):
        _trim_segment(output_pattern % idx, boundaries[idx + 1] - boundaries[idx])


_OPENING_RE = re.compile(r"Opening '(?P<path>.+?)' for writing")
_SEG_INDEX_RE = re.compile(r"(\d+)\.[A-Za-z0-9]+$")
_OUT_TIME_RE = re.compile(r"^out_time_ms=(\d+)")


def _run_chunk_streaming(
    video_path: str,
    output_pattern: str,
    cut_points: list[float],
    start_num: int,
    start_time: float,
    end_time: float | None,
    ffmpeg: str,
    on_segment: Callable[[int, str], None] | None,
    on_progress: Callable[[float], None] | None,
    total_duration: float | None,
    video_args: list[str],
) -> None:
    cmd = [ffmpeg, "-y", "-nostats"]

    if start_time > 0.0:
        cmd += ["-ss", _fmt_ts(start_time)]
    if end_time is not None:
        cmd += ["-to", _fmt_ts(end_time)]

    cmd += [
        "-i", video_path,
        "-map", "0:v:0",
        "-map", "0:a?",
        *video_args,
        "-c:a", "aac",
        "-b:a", "160k",
        "-ac", "2",
        "-ar", "48000",
        "-f", "segment",
        "-segment_times", _fmt_segment_times(cut_points),
        "-segment_start_number", str(start_num),
        "-reset_timestamps", "1",
        "-progress", "pipe:2",
        output_pattern,
    ]

    proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace",
        creationflags=CREATE_NO_WINDOW,
    )

    current: tuple[int, str] | None = None
    tail: list[str] = []
    assert proc.stderr is not None
    for line in proc.stderr:
        line = line.rstrip("\r\n")
        if not line:
            continue
        tail.append(line)
        if len(tail) > 40:
            tail.pop(0)

        m = _OPENING_RE.search(line)
        if m:
            path = m.group("path")
            im = _SEG_INDEX_RE.search(os.path.basename(path))
            if im:
                if current is not None and on_segment:
                    on_segment(current[0], current[1])
                current = (int(im.group(1)), path)
            continue

        tm = _OUT_TIME_RE.match(line)
        if tm and on_progress and total_duration and total_duration > 0:
            secs = start_time + int(tm.group(1)) / 1_000_000.0
            on_progress(min(1.0, secs / total_duration))

    code = proc.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg exit {code}: {os.linesep.join(tail[-15:])}")

    if current is not None and on_segment:
        on_segment(current[0], current[1])


def run_ffmpeg_segment_streaming(
    video_path: str,
    output_pattern: str,
    cut_points: list[float],
    total_duration: float | None = None,
    ffmpeg: str | None = None,
    on_segment: Callable[[int, str], None] | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> None:
    """Same cut as :func:`run_ffmpeg_segment`, with live callbacks.

    ``on_segment(index, path)`` fires as each segment finishes (detected from
    the muxer opening the next output file, plus a final flush at process
    exit) -- each segment is trimmed back to its intended duration (see
    :func:`run_ffmpeg_segment`) before this fires, so callers always see the
    corrected file. ``on_progress(fraction)`` fires from ffmpeg's
    ``-progress`` stream. Identical ffmpeg arguments and output to the
    non-streaming variant, including the stream-copy-then-reencode fallback.
    """
    ff = ffmpeg or get_ffmpeg()
    duration = total_duration if total_duration is not None else probe_video_duration(video_path)
    boundaries = _segment_boundaries(cut_points, duration)

    def _on_segment(idx: int, path: str) -> None:
        if 0 <= idx < len(boundaries) - 1:
            _trim_segment(path, boundaries[idx + 1] - boundaries[idx])
        if on_segment:
            on_segment(idx, path)

    def _run_all(video_args: list[str]) -> None:
        if len(cut_points) <= CHUNK_SIZE:
            _run_chunk_streaming(video_path, output_pattern, cut_points, 0, 0.0,
                                 None, ff, _on_segment, on_progress, duration, video_args)
            return

        for i in range(0, len(cut_points), CHUNK_SIZE):
            chunk = cut_points[i: i + CHUNK_SIZE]
            start_time = cut_points[i - 1] if i > 0 else 0.0
            end_time = chunk[-1] if i + CHUNK_SIZE < len(cut_points) else None
            relative = [p - start_time for p in chunk]
            _run_chunk_streaming(video_path, output_pattern, relative, i, start_time,
                                 end_time, ff, _on_segment, on_progress, duration, video_args)

    try:
        _run_all(_COPY_VIDEO_ARGS)
    except RuntimeError as exc:
        log(f"Stream copy failed ({exc}), re-encoding instead")
        _run_all(_REENCODE_VIDEO_ARGS)


def collect_scenes(
    output_dir: str,
    file_name: str,
    cut_points: list[float],
    total_duration: float,
) -> list[dict[str, Any]]:
    """Build scene metadata list from output directory and cut points.

    Scans ``output_dir`` for ``{file_name}_{index:04d}.mp4`` files and
    builds a dict per scene with timing, path, and thumbnail info.

    Args:
        output_dir: Directory containing segmented clip files.
        file_name: Base name for clips (usually the video stem).
        cut_points: Sorted cut timestamps used for segmentation.
        total_duration: Total video duration in seconds.

    Returns:
        List of scene dicts with keys: ``scene_index``, ``start``,
        ``end``, ``duration``, ``path``, ``thumbnail``, ``original_file``.
    """
    scenes: list[dict[str, Any]] = []
    boundaries = [0.0] + cut_points
    all_boundaries = boundaries + [total_duration]

    for index in range(len(boundaries)):
        start = all_boundaries[index]
        end = all_boundaries[index + 1]
        clip_path = os.path.join(output_dir, f"{file_name}_{index:04d}.mp4")
        thumb_path = os.path.join(output_dir, f"{file_name}_{index:04d}.jpg")

        if os.path.exists(clip_path) and os.path.getsize(clip_path) > 0:
            scenes.append({
                "scene_index": index,
                "start": start,
                "end": end,
                "duration": round(end - start, 3),
                "path": clip_path,
                "thumbnail": thumb_path,
                "original_file": file_name,
            })

    return scenes
