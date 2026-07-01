from __future__ import annotations

import os
import subprocess
import sys
from typing import Callable, Optional, Dict, Tuple

from ..infra.binaries import get_ffmpeg, get_ffprobe

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def dedup_ffmpeg(
    video_path: str,
    output_path: str,
    threshold: float = 2.0,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> Tuple[str, Dict]:
    ffmpeg = get_ffmpeg()
    ffprobe = get_ffprobe()

    def _probe_frames(path):
        try:
            r = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=nb_read_packets",
                 "-of", "default=noprint_wrappers=1:nokey=1", path],
                capture_output=True, text=True, timeout=30,
                creationflags=CREATE_NO_WINDOW,
            )
            val = r.stdout.strip()
            if val.isdigit():
                return int(val)
        except Exception:
            pass

        try:
            r = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "v:0",
                 "-count_frames", "-show_entries", "stream=nb_read_frames",
                 "-of", "default=noprint_wrappers=1:nokey=1", path],
                capture_output=True, text=True, timeout=30,
                creationflags=CREATE_NO_WINDOW,
            )
            val = r.stdout.strip()
            if val.isdigit():
                return int(val)
        except Exception:
            pass
        return 0

    frames_in = _probe_frames(video_path)

    if progress_cb:
        progress_cb(0, "Removing duplicate frames (mpdecimate)...")

    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-vf", f"mpdecimate=hi={int(threshold)}",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path,
    ]

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600,
                           creationflags=CREATE_NO_WINDOW)
        if r.returncode != 0:
            raise RuntimeError(f"FFmpeg mpdecimate failed: {r.stderr.strip()}")
    except subprocess.TimeoutExpired:
        raise RuntimeError("FFmpeg mpdecimate timed out after 1 hour")

    frames_out = _probe_frames(output_path)

    stats = {
        "frames_in": frames_in,
        "frames_out": frames_out,
        "frames_removed": max(0, frames_in - frames_out),
        "pct_removed": round((1 - frames_out / max(1, frames_in)) * 100, 1),
    }

    if progress_cb:
        progress_cb(100, f"Complete ({frames_out}/{frames_in} frames kept, {stats['pct_removed']}% removed)")

    return output_path, stats
