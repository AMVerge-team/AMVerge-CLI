"""Keyframe detection - the default, fastest method.

Splits at I-frame boundaries with lossless stream copy.
No extra dependencies needed.

Usage:
    python 01_basic_detect.py [video_path]

Output:
    <video>_scenes/ with .mp4 clips, .jpg thumbnails, scenes.json
"""

import sys
from pathlib import Path
from amverge import detect_scenes

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "episode.mp4"

result = detect_scenes(
    video_path=VIDEO,
    method="keyframe",
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

print(f"\nscenes.json saved to {result.scenes_json}")
