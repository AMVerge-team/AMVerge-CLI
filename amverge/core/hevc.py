"""HEVC codec detection via ffprobe."""
from __future__ import annotations

import subprocess
import sys

from .binaries import get_ffprobe

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def is_hevc(video_path: str) -> bool:
    """Return True if the first video stream is encoded as HEVC/H.265."""
    if not video_path or not str(video_path).strip():
        raise ValueError("No video path provided")

    ffprobe = get_ffprobe()
    result = subprocess.run(
        [
            ffprobe,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name",
            "-of", "default=nk=1:nw=1",
            video_path,
        ],
        capture_output=True,
        text=True,
        creationflags=CREATE_NO_WINDOW,
    )

    if result.returncode != 0:
        err = (result.stderr or "").strip()
        raise RuntimeError(f"ffprobe failed (exit {result.returncode})" + (f": {err}" if err else ""))

    return result.stdout.strip().lower() == "hevc"
