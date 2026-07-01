from .dedup_ffmpeg import dedup_ffmpeg
from .dedup_ssim import dedup_ssim, SSIM_AVAILABLE
from .dedup_framediff import dedup_framediff, FRAMEDIFF_AVAILABLE
from .dedup_advanced import dedup_advanced, ADVANCED_AVAILABLE

DEDUP_METHODS = {
    "ffmpeg": {
        "name": "mpdecimate (FFmpeg)",
        "description": "Fast FFmpeg mpdecimate filter, no extra deps",
        "requires": None,
    },
    "framediff": {
        "name": "FrameDiff (OpenCV)",
        "description": "Pixel-level motion detection with adaptive threshold",
        "requires": "opencv",
    },
    "ssim": {
        "name": "SSIM (OpenCV)",
        "description": "Structural similarity index, quality-aware",
        "requires": "opencv + scikit-image",
    },
    "advanced": {
        "name": "Advanced (OpenCV)",
        "description": "Optical flow + region grid + camera compensation + static subject detection",
        "requires": "opencv",
    },
}

__all__ = [
    "dedup_ffmpeg",
    "dedup_ssim",
    "dedup_framediff",
    "dedup_advanced",
    "SSIM_AVAILABLE",
    "FRAMEDIFF_AVAILABLE",
    "ADVANCED_AVAILABLE",
    "DEDUP_METHODS",
]
