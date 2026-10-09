"""Keyframe snapping - see how a scene's cut widens to enclosing keyframes.

Shows what cutting.smart_cut.snap_range_to_keyframes does before a "copy"
mode cut: widen [start, end) outward to the last keyframe at or before
start, and the first keyframe at or after end.

Usage:
    python 02_align_scenes.py [video_path]
"""

import sys
from amverge import get_keyframe_timestamps_pyav
from amverge.core.cutting.smart_cut import snap_range_to_keyframes

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "episode.mp4"

# Simulated scene boundaries (you would get these from TransNetV2)
example_scenes = [
    (0.0, 5.0),
    (5.0, 10.0),
    (10.2, 15.0),   # starts 0.2s after a keyframe -> widens back to it
    (15.5, 20.3),   # ends 0.3s after a keyframe -> widens forward past it
]

kf = get_keyframe_timestamps_pyav(VIDEO)
print(f"\n{len(kf)} keyframes extracted from {VIDEO}")
print(f"\n{len(example_scenes)} example scenes:\n")

for start, end in example_scenes:
    clip_start, clip_end = snap_range_to_keyframes(kf, start, end)
    end_desc = f"{clip_end:.2f}s" if clip_end is not None else "EOF"
    widened = clip_start != start or clip_end != end
    print(f"  [{start:6.2f}s, {end:6.2f}s)  ->  [{clip_start:6.2f}s, {end_desc:>6})"
          f"  {'(widened)' if widened else '(already aligned)'}")
