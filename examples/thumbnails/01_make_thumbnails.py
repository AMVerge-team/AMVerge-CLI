"""Batch thumbnail generation.

Generates thumbnails for scene clips using ThreadPoolExecutor.
Each thumbnail is 960px wide progressive JPEG at 95% quality.

Usage:
    python 01_make_thumbnails.py [video_path]
"""

import sys
from pathlib import Path
from amverge import detect_scenes, make_thumbnail, generate_thumbnails

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "episode.mp4"

# Generate one thumbnail manually
result = detect_scenes(VIDEO, method="keyframe", thumbnails=False)
if result.scenes:
    first_clip = result.scenes[0].path
    thumb_path = "example_manual_thumb.jpg"
    ok = make_thumbnail(first_clip, thumb_path)
    print(f"Manual thumbnail: {'OK' if ok else 'FAIL'} -> {thumb_path}")

# Or generate all thumbnails at once (scenes dict format)
scene_dicts = [
    {"scene_index": s.index, "path": s.path}
    for s in result.scenes[:3]
]
out_dir = "example_thumbs"
generate_thumbnails(scene_dicts, out_dir, Path(VIDEO).stem, workers=4)
print(f"\nBatch thumbnails: {out_dir}/")
for p in sorted(Path(out_dir).glob("*.jpg")):
    print(f"  {p.name}  ({p.stat().st_size:,} bytes)")
