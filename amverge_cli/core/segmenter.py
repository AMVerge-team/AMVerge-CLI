"""FFmpeg segment-based scene cutting."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from .binaries import get_ffmpeg

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

_SILENCED = [
    re.compile(r"track\s+\d+:\s+codec frame size is not set", re.IGNORECASE),
    re.compile(r"^\[segment\s+@\s+[^\]]+\]\s+Opening\s+'.+'\s+for writing$", re.IGNORECASE),
]

CHUNK_SIZE = 1500


def _fmt_ts(seconds: float) -> str:
    return f"{float(seconds):.6f}".rstrip("0").rstrip(".")


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
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "160k",
        "-ac", "2",
        "-ar", "48000",
        "-f", "segment",
        "-segment_times", ",".join(_fmt_ts(p) for p in cut_points),
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
) -> None:
    ff = ffmpeg or get_ffmpeg()

    if len(cut_points) <= CHUNK_SIZE:
        _run_chunk(video_path, output_pattern, cut_points, 0, 0.0, None, ff)
        return

    for i in range(0, len(cut_points), CHUNK_SIZE):
        chunk = cut_points[i: i + CHUNK_SIZE]
        start_time = cut_points[i - 1] if i > 0 else 0.0
        end_time = chunk[-1] if i + CHUNK_SIZE < len(cut_points) else None
        relative = [p - start_time for p in chunk]
        _run_chunk(video_path, output_pattern, relative, i, start_time, end_time, ff)


def collect_scenes(
    output_dir: str,
    file_name: str,
    cut_points: list[float],
    total_duration: float,
) -> list[dict[str, Any]]:
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
