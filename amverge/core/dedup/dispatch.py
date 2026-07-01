from __future__ import annotations

from typing import Callable, Optional, Tuple

from ._encode import build_stats, encode_selected, export_frame_list

DEFAULT_THRESHOLD = {"ffmpeg": 0.33, "ssim": 0.987, "framediff": 10.0, "advanced": 1.0}


def run_dedup(
    video_path: str,
    output_path: str,
    method: str = "ffmpeg",
    threshold: Optional[float] = None,
    min_change_pct: float = 2.0,
    codec: Optional[str] = None,
    crf: int = 18,
    dry_run: bool = False,
    export_frames: Optional[str] = None,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> Tuple[Optional[str], dict]:
    """Unified dedup entry point.

    Analysis methods (ssim/framediff/advanced) support ``dry_run`` (analyze
    only, no encode) and ``export_frames`` (write kept/removed ranges to CSV).
    The ffmpeg (mpdecimate) method decides frames inside the native filter and
    cannot enumerate them, so it rejects those options.

    Returns (output_path, stats); output_path is None on a dry run.
    """
    if threshold is None:
        threshold = DEFAULT_THRESHOLD.get(method, 0.0)

    if method == "ffmpeg":
        if dry_run or export_frames:
            raise ValueError(
                "The ffmpeg method decides frames inside mpdecimate and cannot "
                "enumerate them. Use ssim/framediff/advanced for --dry-run or "
                "--export-frames."
            )
        from .dedup_ffmpeg import dedup_ffmpeg
        return dedup_ffmpeg(video_path, output_path, threshold, progress_cb,
                            codec=codec, crf=crf)

    if method == "ssim":
        from .dedup_ssim import analyze_ssim
        keep, frames_in, fps = analyze_ssim(video_path, threshold, progress_cb)
        cadence = {}
    elif method == "framediff":
        from .dedup_framediff import analyze_framediff
        keep, frames_in, fps = analyze_framediff(
            video_path, threshold, min_change_pct, progress_cb)
        cadence = {}
    elif method == "advanced":
        from .dedup_advanced import analyze_advanced
        keep, frames_in, fps, cadence = analyze_advanced(
            video_path, threshold, progress_cb)
    else:
        raise ValueError(f"Unknown dedup method '{method}'")

    stats = build_stats(frames_in, len(keep))
    stats.update(cadence)

    if export_frames:
        export_frame_list(export_frames, keep, frames_in, fps)

    if dry_run:
        return None, stats

    if progress_cb:
        progress_cb(95, "Encoding output...")
    encode_selected(video_path, output_path, keep, crf=crf, codec=codec,
                    progress_cb=progress_cb, progress_lo=95, progress_hi=99)
    if progress_cb:
        progress_cb(100, f"Complete ({len(keep)}/{frames_in} frames kept)")

    return output_path, stats
