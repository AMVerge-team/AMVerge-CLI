from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

from ._encode import build_stats, encode_selected, export_frame_list
from .dedup_ssim import SSIM_AVAILABLE
from .dedup_advanced import ADVANCED_AVAILABLE

DEFAULT_THRESHOLD = {"ffmpeg": 0.33, "ssim": 0.987, "framediff": 10.0, "advanced": 1.0}

PRESETS: Dict[str, Dict[str, dict]] = {
    "aggressive": {
        "ffmpeg":   {"threshold": 0.25, "hi": 400, "lo": 200},
        "ssim":     {"threshold": 0.995, "ssim_window": 5},
        "framediff":{"threshold": 5.0, "min_change_pct": 1.0},
        "advanced": {"threshold": 0.6, "cadence_gate": True},
    },
    "normal": {
        "ffmpeg":   {"threshold": 0.33, "hi": 768, "lo": 320},
        "ssim":     {"threshold": 0.987, "ssim_window": 3},
        "framediff":{"threshold": 10.0, "min_change_pct": 2.0},
        "advanced": {"threshold": 1.0, "cadence_gate": True},
    },
    "gentle": {
        "ffmpeg":   {"threshold": 0.45, "hi": 1200, "lo": 500},
        "ssim":     {"threshold": 0.975, "ssim_window": 2},
        "framediff":{"threshold": 18.0, "min_change_pct": 4.0},
        "advanced": {"threshold": 1.8, "cadence_gate": False},
    },
}

PRESET_LABELS = {
    "aggressive": "Aggressive — removes more frames, slightly riskier (best for clean animation)",
    "normal":     "Normal — balanced, safe defaults",
    "gentle":      "Gentle — removes fewer frames, safest (best for grainy or live-action)",
}


def auto_detect_method() -> str:
    if ADVANCED_AVAILABLE:
        return "advanced"
    if SSIM_AVAILABLE:
        return "ssim"
    return "ffmpeg"


def resolve_preset(preset: str, method: str) -> dict:
    p = PRESETS.get(preset, PRESETS["normal"])
    return p.get(method, p.get("ffmpeg", {}))


def run_dedup_simple(
    video_path: str,
    output_path: str,
    preset: str = "normal",
    codec: Optional[str] = None,
    crf: int = 18,
    dry_run: bool = False,
    export_frames: Optional[str] = None,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> Tuple[Optional[str], dict]:
    method = auto_detect_method()
    overrides = resolve_preset(preset, method)
    threshold = overrides.get("threshold", DEFAULT_THRESHOLD.get(method, 0.0))
    hi = overrides.get("hi", 768)
    lo = overrides.get("lo", 320)
    ssim_window = overrides.get("ssim_window", 3)
    cadence_gate = overrides.get("cadence_gate", True)
    min_change_pct = overrides.get("min_change_pct", 2.0)

    return run_dedup(
        video_path, output_path,
        method=method,
        threshold=threshold,
        min_change_pct=min_change_pct,
        codec=codec, crf=crf,
        dry_run=dry_run,
        export_frames=export_frames,
        progress_cb=progress_cb,
        hi=hi, lo=lo,
        ssim_window=ssim_window,
        cadence_gate=cadence_gate,
    )


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
    hi: int = 768,
    lo: int = 320,
    ssim_window: int = 3,
    cadence_gate: bool = True,
) -> Tuple[Optional[str], dict]:
    """Unified dedup entry point.

    Analysis methods (ssim/framediff/advanced) support ``dry_run`` (analyze
    only, no encode) and ``export_frames`` (write kept/removed ranges to CSV).
    The ffmpeg (mpdecimate) method decides frames inside the native filter and
    cannot enumerate them, so it rejects those options.

    Args:
        hi, lo: mpdecimate per-block SAD thresholds (ffmpeg method only).
        ssim_window: number of recent kept frames to compare against (ssim only).
        cadence_gate: remove false-keeps when cadence confidently detected
            (advanced method only).

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
        return dedup_ffmpeg(video_path, output_path, threshold, hi, lo,
                            progress_cb, codec=codec, crf=crf)

    if method == "ssim":
        from .dedup_ssim import analyze_ssim
        keep, frames_in, fps = analyze_ssim(
            video_path, threshold, window_size=ssim_window, progress_cb=progress_cb)
        cadence = {}
    elif method == "framediff":
        from .dedup_framediff import analyze_framediff
        keep, frames_in, fps = analyze_framediff(
            video_path, threshold, min_change_pct, progress_cb)
        cadence = {}
    elif method == "advanced":
        from .dedup_advanced import analyze_advanced
        keep, frames_in, fps, cadence = analyze_advanced(
            video_path, threshold, cadence_gate=cadence_gate, progress_cb=progress_cb)
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
