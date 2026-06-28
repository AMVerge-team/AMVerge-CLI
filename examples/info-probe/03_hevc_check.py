"""HEVC codec check.

Quick ffprobe-based check for HEVC/H.265 encoding status.

Usage:
    python 03_hevc_check.py [video_path]
"""

import sys
from amverge import check_if_hevc

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "episode.mp4"

if check_if_hevc(VIDEO):
    print(f"{VIDEO} is HEVC (H.265)")
    print("Smart cut will use snapped_copy on CPU or full re-encode with GPU")
else:
    print(f"{VIDEO} is H.264 or other")
    print("Smart cut: lossless copy when keyframe-aligned, smartcut otherwise")
