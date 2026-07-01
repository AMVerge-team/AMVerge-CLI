from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
from typing import Callable, List, Optional, Sequence, Tuple

from ..infra.binaries import get_ffmpeg, get_ffprobe
from ..upscaling.ffmpeg_helpers import get_color_args

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def probe_frame_count(video_path: str) -> int:
    ffprobe = get_ffprobe()
    cmd = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-count_packets", "-show_entries", "stream=nb_read_packets",
        "-of", "csv=p=0", video_path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                           creationflags=CREATE_NO_WINDOW)
        return int((r.stdout or "0").strip() or 0)
    except Exception:
        return 0


def probe_duration(video_path: str) -> float:
    ffprobe = get_ffprobe()
    cmd = [
        ffprobe, "-v", "error", "-show_entries", "format=duration",
        "-of", "csv=p=0", video_path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                           creationflags=CREATE_NO_WINDOW)
        return float((r.stdout or "0").strip() or 0.0)
    except Exception:
        return 0.0


def _probe_field(video_path: str, entry: str) -> str:
    ffprobe = get_ffprobe()
    cmd = [
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries", f"stream={entry}", "-of", "csv=p=0", video_path,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                           creationflags=CREATE_NO_WINDOW)
        return (r.stdout or "").strip().lower()
    except Exception:
        return ""


def is_interlaced(video_path: str) -> bool:
    return _probe_field(video_path, "field_order") in ("tt", "bb", "tb", "bt")


def pick_pixfmt_profile(video_path: str) -> Tuple[str, str]:
    """Return (pix_fmt, x264 profile) preserving source bit depth (8 vs 10)."""
    pix = _probe_field(video_path, "pix_fmt")
    if "10" in pix or "12" in pix or pix.startswith("p010"):
        return "yuv420p10le", "high10"
    return "yuv420p", "high"


def build_stats(frames_in: int, frames_out: int) -> dict:
    removed = max(0, frames_in - frames_out)
    pct = (removed / frames_in * 100.0) if frames_in > 0 else 0.0
    return {
        "frames_in": frames_in,
        "frames_out": frames_out,
        "frames_removed": removed,
        "pct_removed": round(pct, 2),
    }


def run_ffmpeg_progress(
    cmd: Sequence[str],
    duration: float,
    progress_cb: Optional[Callable[[int, str], None]],
    lo: int = 0,
    hi: int = 100,
    label: str = "Encoding",
    timeout: float = 7200.0,
) -> None:
    """Run ffmpeg with -progress on stdout, mapping time into [lo, hi].

    A watchdog kills the process if it exceeds ``timeout`` seconds (a stalled
    ffmpeg would otherwise block the stdout read loop forever).
    """
    proc = subprocess.Popen(
        list(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, creationflags=CREATE_NO_WINDOW,
    )
    timed_out = {"v": False}

    def _kill():
        timed_out["v"] = True
        proc.kill()

    watchdog = threading.Timer(timeout, _kill)
    watchdog.start()
    last = -1
    try:
        for line in proc.stdout:
            if progress_cb and duration > 0 and line.startswith("out_time_us="):
                try:
                    secs = int(line.split("=", 1)[1].strip()) / 1_000_000.0
                except ValueError:
                    continue
                pct = lo + int(min(1.0, secs / duration) * (hi - lo))
                if pct != last:
                    progress_cb(min(hi, pct), label)
                    last = pct
    finally:
        watchdog.cancel()
        proc.stdout.close()
        stderr = proc.stderr.read()
        proc.stderr.close()
        rc = proc.wait()
    if timed_out["v"]:
        raise RuntimeError(f"FFmpeg timed out after {int(timeout)}s")
    if rc != 0:
        raise RuntimeError(f"FFmpeg failed: {stderr.strip()}")


def _ranges(indices: Sequence[int]) -> List[Tuple[int, int]]:
    ranges: List[Tuple[int, int]] = []
    start = prev = None
    for i in sorted(set(indices)):
        if start is None:
            start = prev = i
        elif i == prev + 1:
            prev = i
        else:
            ranges.append((start, prev))
            start = prev = i
    if start is not None:
        ranges.append((start, prev))
    return ranges


def _build_select_expr(keep_indices: Sequence[int]) -> str:
    terms = []
    for a, b in _ranges(keep_indices):
        if a == b:
            terms.append(f"eq(n\\,{a})")
        else:
            terms.append(f"between(n\\,{a}\\,{b})")
    return "+".join(terms) if terms else "0"


def encode_selected(
    video_path: str,
    output_path: str,
    keep_indices: Sequence[int],
    crf: int = 18,
    preset: str = "fast",
    progress_cb: Optional[Callable[[int, str], None]] = None,
    progress_lo: int = 0,
    progress_hi: int = 100,
) -> str:
    """Re-encode keeping only frames in keep_indices, preserving audio, color
    metadata and source bit depth. Interlaced input is deinterlaced (yadif=0,
    1:1 so frame indices stay aligned). Output is VFR: kept frames retain
    source timestamps, so duration is unchanged and copied audio stays synced.

    Uses a filter_complex_script file to avoid the Windows command-line length
    limit for large frame lists.
    """
    ffmpeg = get_ffmpeg()
    expr = _build_select_expr(keep_indices)
    pre = "yadif=0," if is_interlaced(video_path) else ""
    graph = f"[0:v]{pre}select='{expr}'[v]"
    pix_fmt, profile = pick_pixfmt_profile(video_path)

    fd, script = tempfile.mkstemp(suffix=".txt", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(graph)
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-progress", "pipe:1", "-nostats",
            "-i", video_path,
            "-filter_complex_script", script,
            "-map", "[v]", "-map", "0:a?",
            "-c:v", "libx264", "-crf", str(int(crf)), "-preset", preset,
            "-profile:v", profile, "-pix_fmt", pix_fmt,
            "-c:a", "copy",
            "-fps_mode", "vfr",
            *get_color_args(video_path),
            "-movflags", "+faststart",
            output_path,
        ]
        run_ffmpeg_progress(cmd, probe_duration(video_path), progress_cb,
                            progress_lo, progress_hi, "Encoding output...")
    finally:
        try:
            os.unlink(script)
        except OSError:
            pass

    return output_path
