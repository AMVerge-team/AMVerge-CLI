"""Keyframe detection - the default, fastest method.

Splits at I-frame boundaries with lossless stream copy.

Usage:
    python 01_basic_detect.py [video_path]
"""

import sys
from amverge import AmvergeVideo

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "episode.mp4"

video = AmvergeVideo(VIDEO)
print(f"{video.name}: {video.width}x{video.height} {video.fps}fps {video.duration:.1f}s\n")

result = video.detect_scenes(
    method="keyframe",
    min_duration=0.25,
    thumbnails=True,
    similarity=True,
    progress=lambda stage, pct, msg: print(f"  [{stage}] {pct}% {msg}"),
)

print(f"\n{len(result.scenes)} scenes detected")
for scene in result.scenes[:10]:
    print(f"  [{scene.index:02d}] {scene.start:6.2f}s - {scene.end:6.2f}s  "
          f"({scene.duration:5.2f}s)  {Path(scene.path).name}")

if len(result.scenes) > 10:
    print(f"  ... and {len(result.scenes) - 10} more")

if result.similar_pairs:
    print(f"\n{len(result.similar_pairs)} similar pairs found:")
    for a, b in result.similar_pairs:
        print(f"  scenes {a} and {b} look similar")
