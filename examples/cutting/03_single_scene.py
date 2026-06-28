"""Single scene cut - inspect which mode is used.

Calls cut_scene directly for one time range. Shows the automatic
mode selection logic in action.

Usage:
    python 03_single_scene.py [video_path]
"""

import sys
from pathlib import Path
from amverge import cut_scene, get_keyframe_timestamps_pyav, check_if_hevc

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "episode.mp4"

kf = get_keyframe_timestamps_pyav(VIDEO)
is_hevc = check_if_hevc(VIDEO)
out_dir = Path("examples_single_cut")
out_dir.mkdir(parents=True, exist_ok=True)

print(f"Source: {Path(VIDEO).name}  HEVC={is_hevc}  {len(kf)} keyframes\n")

# Test: scene starting exactly on a keyframe -> copy
for start, end, desc in [
    (0.0, 5.0, "starts on keyframe 0.0s -> should be copy"),
    (0.3, 5.0, "starts between keyframes -> may need smartcut/reencode"),
]:
    path, mode = cut_scene(Path(VIDEO), start, end, 0, out_dir, kf, False, is_hevc)
    print(f"  {start:.1f}s - {end:.1f}s  ({desc})")
    print(f"  Result: {mode} -> {Path(path).name}\n")
