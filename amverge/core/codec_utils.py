from __future__ import annotations

"""Codec detection utilities and export codec profiles.

Contains HEVC detection helpers and the full codec/audio/container mapping
used by the export command and wizard. Library users can import these to
build custom export pipelines.

Example:
    >>> from amverge import CODEC_PROFILES, AUDIO_FFMPEG, check_if_hevc
    >>> check_if_hevc("episode.mp4")
    False
    >>> CODEC_PROFILES["h264_main"]["cpu"]
    'libx264'
"""

import subprocess
from pathlib import Path

from .binaries import get_ffprobe

# -- Codec profile tables --------------------------------------------------

VALID_CODECS = {
    "copy",
    "h264", "hevc", "h265",
    "h264_main", "h264_high", "h264_high10", "h264_high422",
    "h265_main", "h265_main10", "h265_main12", "h265_main422_10",
    "av1_main",
    "prores_422_lt", "prores_422", "prores_422_hq", "prores_4444", "prores_4444_xq",
}
VALID_AUDIO = {"copy", "aac", "aac_320", "pcm16", "pcm24", "flac", "alac", "opus", "mp3", "none"}
VALID_CONTAINERS = {"mp4", "mkv", "mov"}
VALID_HARDWARE = {"auto", "gpu", "cpu"}

CODEC_ALIASES: dict[str, str] = {
    "h264": "h264_main",
    "hevc": "h265_main",
    "h265": "h265_main",
}

CODEC_PROFILES: dict[str, dict[str, str | None]] = {
    "h264_main":       {"cpu": "libx264",   "gpu": "h264_nvenc",      "args": "-profile:v main"},
    "h264_high":       {"cpu": "libx264",   "gpu": "h264_nvenc",      "args": "-profile:v high"},
    "h264_high10":     {"cpu": "libx264",   "gpu": None,              "args": "-profile:v high10"},
    "h264_high422":    {"cpu": "libx264",   "gpu": None,              "args": "-profile:v high422"},
    "h265_main":       {"cpu": "libx265",   "gpu": "hevc_nvenc",      "args": "-profile:v main"},
    "h265_main10":     {"cpu": "libx265",   "gpu": "hevc_nvenc",      "args": "-profile:v main10"},
    "h265_main12":     {"cpu": "libx265",   "gpu": None,              "args": "-profile:v main12"},
    "h265_main422_10": {"cpu": "libx265",   "gpu": None,              "args": "-profile:v main422-10"},
    "av1_main":        {"cpu": "libsvtav1", "gpu": "av1_nvenc",       "args": ""},
    "prores_422_lt":   {"cpu": "prores_ks", "gpu": None,              "args": "-profile:v 0"},
    "prores_422":      {"cpu": "prores_ks", "gpu": None,              "args": "-profile:v 1"},
    "prores_422_hq":   {"cpu": "prores_ks", "gpu": None,              "args": "-profile:v 2"},
    "prores_4444":     {"cpu": "prores_ks", "gpu": None,              "args": "-profile:v 3"},
    "prores_4444_xq":  {"cpu": "prores_ks", "gpu": None,              "args": "-profile:v 4"},
}

PRORES_CODECS = {k for k in CODEC_PROFILES if k.startswith("prores")}

AUDIO_FFMPEG: dict[str, list[str]] = {
    "copy":     ["-c:a", "copy"],
    "aac":      ["-c:a", "aac"],
    "aac_320":  ["-c:a", "aac", "-b:a", "320k"],
    "pcm16":    ["-c:a", "pcm_s16le"],
    "pcm24":    ["-c:a", "pcm_s24le"],
    "flac":     ["-c:a", "flac"],
    "alac":     ["-c:a", "alac"],
    "opus":     ["-c:a", "libopus"],
    "mp3":      ["-c:a", "libmp3lame"],
    "none":     ["-an"],
}


def _resolve_gpu(hardware: str, codec: str) -> bool:
    if codec == "copy":
        return False
    if codec in PRORES_CODECS:
        return False
    if hardware == "cpu":
        return False
    if hardware == "gpu":
        return True
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def check_if_hevc(video: str | Path) -> bool:
    """Check if a video file is HEVC (H.265) encoded.

    Uses ffprobe to probe the first video stream's codec name.

    Args:
        video: Path to the video file.

    Returns:
        True if the video codec is ``"hevc"``, False otherwise.

    Raises:
        ValueError: If ``video`` is an empty string.
        RuntimeError: If ffprobe exits with a non-zero code.

    Example:
        >>> check_if_hevc("episode.mp4")
        False
    """
    path = str(video)
    if not path.strip():
        raise ValueError("No video path provided")

    cmd = [
        get_ffprobe(),
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name",
        "-of", "default=nk=1:nw=1",
        path,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        err = (p.stderr or "").strip()
        raise RuntimeError(
            f"ffprobe failed (exit {p.returncode})" + (f": {err}" if err else "")
        )
    return p.stdout.strip().lower() == "hevc"
