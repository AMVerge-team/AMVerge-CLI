from .dedup_ffmpeg import dedup_ffmpeg
from .dedup_ssim import dedup_ssim, SSIM_AVAILABLE
from .dedup_framediff import dedup_framediff, FRAMEDIFF_AVAILABLE
from ._encode import encode_selected, probe_frame_count, build_stats

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
}

__all__ = [
    "dedup_ffmpeg",
    "dedup_ssim",
    "dedup_framediff",
    "encode_selected",
    "probe_frame_count",
    "build_stats",
    "SSIM_AVAILABLE",
    "FRAMEDIFF_AVAILABLE",
    "DEDUP_METHODS",
]
