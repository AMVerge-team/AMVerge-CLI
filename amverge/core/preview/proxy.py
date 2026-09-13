"""Browser-playable preview proxies.

A cut clip keeps whatever codec its source used, because cutting stream-copies
the video. Plenty of sources are HEVC or 10-bit H.264, and a Chromium-based
viewer can decode neither: it demuxes the file and plays the audio while the
picture stays black, with no error anywhere.

This module answers "can a browser show this?" and, when the answer is no,
transcodes a small H.264 copy that it can.
"""

from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional

from ..infra.binaries import get_ffmpeg, get_ffprobe

# What a Chromium-based viewer can decode. HEVC is absent on purpose: the
# bundled builds carry no licence for it.
PLAYABLE_CODECS = {"h264", "vp8", "vp9", "av1", "theora"}

# 8-bit only. H.264 High 10 reports as `h264` and still will not decode, which
# makes the codec name alone an unreliable test.
PLAYABLE_PIX_FMTS = {"yuv420p", "yuvj420p", "yuv422p", "yuvj422p", "rgb24", "bgr24"}

DEFAULT_HEIGHT = 480
DEFAULT_CRF = 30

ProgressCallback = Callable[[float], None]


def probe_video(path: str | Path) -> dict:
    """Codec, profile and pixel format of the first video stream."""
    result = subprocess.run(
        [
            get_ffprobe(), "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,profile,pix_fmt,width,height",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return {}

    try:
        streams = json.loads(result.stdout).get("streams", [])
    except json.JSONDecodeError:
        return {}
    return streams[0] if streams else {}


def is_browser_playable(path: str | Path) -> bool:
    """Whether a viewer can decode this file's video as-is.

    An unreadable file counts as playable so a probe failure does not send
    every clip through a pointless transcode.
    """
    info = probe_video(path)
    if not info:
        return True

    codec = (info.get("codec_name") or "").lower()
    pix_fmt = (info.get("pix_fmt") or "").lower()

    if codec not in PLAYABLE_CODECS:
        return False
    # catches High 10 and any other >8-bit format under a playable codec name
    return pix_fmt in PLAYABLE_PIX_FMTS


def proxy_path_for(clip: str | Path, height: int = DEFAULT_HEIGHT, crf: int = DEFAULT_CRF) -> Path:
    """Where a clip's proxy lives.

    Beside the clip and keyed on the settings that produced it, so changing the
    height or quality makes a new file rather than reusing a stale one.
    """
    p = Path(clip)
    return p.with_name(f"{p.stem}.x264_{height}p{crf}.preview.mp4")


def ensure_preview_proxy(
    clip: str | Path,
    *,
    height: int = DEFAULT_HEIGHT,
    crf: int = DEFAULT_CRF,
    force: bool = False,
    on_progress: Optional[ProgressCallback] = None,
) -> tuple[Path, bool]:
    """Return a playable path for `clip`, transcoding only when necessary.

    Returns ``(path, transcoded)``. `path` is the original when it was already
    playable, so callers can use the result unconditionally.
    """
    source = Path(clip)
    if not source.is_file():
        raise FileNotFoundError(f"No such clip: {source}")

    if not force and is_browser_playable(source):
        return source, False

    target = proxy_path_for(source, height, crf)
    if target.is_file() and target.stat().st_size > 0:
        return target, False

    # temp-then-rename so an interrupted run leaves no half-file; pid keeps two
    # processes asked for the same clip out of each other's way
    tmp = target.with_suffix(f".{os.getpid()}.tmp.mp4")

    command = [
        get_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(source),
        "-c:v", "libx264",
        # keeps the aspect and never upscales a clip smaller than the target
        "-vf", f"scale=-2:'min({height},ih)'",
        "-preset", "veryfast",
        "-crf", str(crf),
        # 8-bit, which is the point of the exercise
        "-pix_fmt", "yuv420p",
        # moves the index to the front so playback can start before the whole
        # file is read
        "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", "128k",
        str(tmp),
    ]

    if on_progress:
        on_progress(0.0)

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(result.stderr.strip() or "ffmpeg failed to build the preview proxy")

    tmp.replace(target)
    if on_progress:
        on_progress(100.0)

    return target, True


def default_jobs() -> int:
    """How many proxies to build at once.

    ffmpeg threads its own encode, so one worker per core spends more time
    contending than encoding. Half the cores, capped, drains a queue fast while
    leaving the machine usable.
    """
    cores = os.cpu_count() or 2
    return max(1, min(4, cores // 2))


def ensure_preview_proxies(
    clips: Iterable[str | Path],
    *,
    height: int = DEFAULT_HEIGHT,
    crf: int = DEFAULT_CRF,
    force: bool = False,
    jobs: Optional[int] = None,
) -> Iterator[tuple[Path, Optional[Path], bool, Optional[str]]]:
    """Build proxies for many clips, yielding each one the moment it lands.

    Results come out of order on purpose, so a caller can use the first
    finished proxy instead of waiting on the slowest. One failure does not stop
    the rest: it is reported as its own result.

    Yields ``(clip, path, transcoded, error)``, where `path` is None only when
    `error` is set.
    """
    sources = [Path(c) for c in clips]
    if not sources:
        return

    with ThreadPoolExecutor(max_workers=jobs or default_jobs()) as pool:
        pending = {
            pool.submit(ensure_preview_proxy, c, height=height, crf=crf, force=force): c
            for c in sources
        }
        for future in as_completed(pending):
            clip = pending[future]
            try:
                path, transcoded = future.result()
                yield clip, path, transcoded, None
            except Exception as exc:
                yield clip, None, False, str(exc)
