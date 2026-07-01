from .dedup_ffmpeg import dedup_ffmpeg
from .dedup_ssim import dedup_ssim, analyze_ssim, SSIM_AVAILABLE
from .dedup_framediff import dedup_framediff, analyze_framediff, FRAMEDIFF_AVAILABLE
from .dedup_advanced import (
    dedup_advanced,
    analyze_advanced,
    detect_cadence,
    ADVANCED_AVAILABLE,
)
from .dispatch import run_dedup
from ._encode import encode_selected, probe_frame_count, build_stats, export_frame_list

DEDUP_METHODS = {
    "ffmpeg": {
        "name": "mpdecimate (FFmpeg)",
        "description": "Fast FFmpeg mpdecimate filter, no extra deps, keeps audio",
        "requires": None,
    },
    "ssim": {
        "name": "SSIM (OpenCV)",
        "description": "Windowed structural similarity, quality-aware",
        "requires": "opencv",
    },
    "framediff": {
        "name": "FrameDiff (OpenCV)",
        "description": "Pixel-level motion detection with adaptive threshold",
        "requires": "opencv",
    },
    "advanced": {
        "name": "Advanced (OpenCV)",
        "description": "Region grid + optical flow + edges + cadence, dead-frame aware",
        "requires": "opencv",
    },
}

__all__ = [
    "dedup_ffmpeg",
    "dedup_ssim",
    "dedup_framediff",
    "dedup_advanced",
    "analyze_ssim",
    "analyze_framediff",
    "analyze_advanced",
    "detect_cadence",
    "run_dedup",
    "encode_selected",
    "probe_frame_count",
    "build_stats",
    "export_frame_list",
    "SSIM_AVAILABLE",
    "FRAMEDIFF_AVAILABLE",
    "ADVANCED_AVAILABLE",
    "DEDUP_METHODS",
]
