from __future__ import annotations

from typing import Callable, Optional, Tuple

from ..infra.binaries import get_ffmpeg
from ..upscaling.ffmpeg_helpers import get_color_args
from ._encode import (
    build_stats,
    is_interlaced,
    pick_pixfmt_profile,
    probe_duration,
    probe_frame_count,
    run_ffmpeg_progress,
)


def dedup_ffmpeg(
    video_path: str,
    output_path: str,
    threshold: float = 0.33,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> Tuple[str, dict]:
    """Remove duplicate frames using FFmpeg mpdecimate filter.

    Args:
        video_path: Path to input video.
        output_path: Path for output video.
        threshold: mpdecimate frac (0-1) - fraction of 8x8 blocks that must
            exceed the change threshold for a frame to be kept. Lower drops
            more; higher keeps more. Default 0.33.
        progress_cb: Optional (pct, msg) callback.

    Returns:
        (output_path, stats) where stats has frames_in/out/removed/pct_removed.
    """
    ffmpeg = get_ffmpeg()

    if progress_cb:
        progress_cb(0, "Removing duplicate frames (mpdecimate)...")

    frac = min(1.0, max(0.0, float(threshold)))
    pre = "yadif=0," if is_interlaced(video_path) else ""
    pix_fmt, profile = pick_pixfmt_profile(video_path)

    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-progress", "pipe:1", "-nostats",
        "-i", video_path,
        "-vf", f"{pre}mpdecimate=hi=768:lo=320:frac={frac}",
        "-fps_mode", "vfr",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-profile:v", profile, "-pix_fmt", pix_fmt,
        "-c:a", "copy",
        *get_color_args(video_path),
        "-movflags", "+faststart",
        output_path,
    ]

    run_ffmpeg_progress(cmd, probe_duration(video_path), progress_cb, 0, 99,
                        "Removing duplicate frames (mpdecimate)...")

    frames_in = probe_frame_count(video_path)
    frames_out = probe_frame_count(output_path)
    stats = build_stats(frames_in, frames_out)

    if progress_cb:
        progress_cb(100, "Complete")

    return output_path, stats
