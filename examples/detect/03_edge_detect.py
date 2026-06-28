"""Edge detection - Canny edges + cosine similarity.

Decodes every frame and compares edge density between adjacent frames.
More accurate than keyframe, slower. Requires amverge[edge].

Usage:
    pip install amverge[edge]
    python 03_edge_detect.py [video_path]
"""

import sys
from pathlib import Path
from amverge import detect_scenes

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "episode.mp4"

result = detect_scenes(
    video_path=VIDEO,
    method="edge",
    min_duration=0.5,
    edge_threshold=0.15,
    edge_radius=0.6,
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
        print(f"  scenes {a} and {b} look similar - consider merging")
