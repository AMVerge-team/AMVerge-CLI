"""Single scene cut - inspect how far a copy widens past the true boundary.

Calls cut_scene directly for one time range in "copy" mode. Shows the
outward keyframe snap and the resulting poster_offset_sec (how far into
the clip the true scene start actually sits).

Usage:
    python 03_single_scene.py [video_path]
"""

import sys
from pathlib import Path
from amverge import cut_scene, get_keyframe_timestamps_pyav

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "episode.mp4"

kf = get_keyframe_timestamps_pyav(VIDEO)
out_dir = Path("examples_single_cut")
out_dir.mkdir(parents=True, exist_ok=True)

print(f"Source: {Path(VIDEO).name}  {len(kf)} keyframes\n")

for start, end, desc in [
    (0.0, 5.0, "starts on keyframe 0.0s -> no widening needed"),
    (0.3, 5.0, "starts between keyframes -> widens back to the preceding one"),
]:
    path, mode, poster_offset_sec = cut_scene(Path(VIDEO), start, end, 0, out_dir, kf, "copy")
    print(f"  {start:.1f}s - {end:.1f}s  ({desc})")
    print(f"  Result: {mode} -> {Path(path).name}  poster_offset_sec={poster_offset_sec:.3f}\n")
