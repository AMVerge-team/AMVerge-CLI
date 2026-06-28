"""TransNetV2 ML detection - best accuracy.

Uses a deep learning model trained for shot boundary detection.
GPU-accelerated when CUDA is available. Requires amverge[ml].

Smart cut pipeline: lossless copy for keyframe-aligned scenes,
smartcut or re-encode for the rest.

Usage:
    pip install amverge[ml]
    python 02_transnetv2_detect.py [video_path]
"""

import sys
from pathlib import Path
from amverge import detect_scenes, TRANSNET_AVAILABLE

if not TRANSNET_AVAILABLE:
    print("Error: transnetv2_pytorch not installed.")
    print("Run: pip install amverge[ml]")
    sys.exit(1)

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "episode.mp4"

result = detect_scenes(
    video_path=VIDEO,
    method="transnetv2",
    min_duration=0.25,
    thumbnails=True,
    similarity=True,
    progress=lambda stage, pct, msg: print(f"  [{stage}] {pct}% {msg}"),
)

print(f"\n{len(result.scenes)} scenes detected in {Path(VIDEO).name}")
for scene in result.scenes:
    print(f"  [{scene.index:02d}] {scene.start:6.2f}s - {scene.end:6.2f}s  "
          f"({scene.duration:5.2f}s)  {Path(scene.path).name}")

if result.similar_pairs:
    print(f"\n{len(result.similar_pairs)} similar pairs found:")
    for a, b in result.similar_pairs:
        print(f"  scenes {a} and {b} look similar")
