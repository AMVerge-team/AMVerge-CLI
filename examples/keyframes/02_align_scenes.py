"""Keyframe alignment - classify scenes for cutting strategy.

Shows which scenes can be losslessly copied vs. need smartcut/re-encode.

Usage:
    python 02_align_scenes.py [video_path]
"""

import sys
from amverge import get_keyframe_timestamps_pyav, classify_scenes_by_keyframe_alignment

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "episode.mp4"

# Simulated scene boundaries (you would get these from TransNetV2)
example_scenes = [
    (0.0, 5.0),
    (5.0, 10.0),
    (10.2, 15.0),   # starts 0.2s after keyframe at 10.0 - still aligned
    (15.5, 20.0),   # starts 0.5s after keyframe at 15.0 - re-encode
]

kf = get_keyframe_timestamps_pyav(VIDEO)
copy, reencode = classify_scenes_by_keyframe_alignment(example_scenes, kf, threshold=0.2)

print(f"\n{len(kf)} keyframes extracted from {VIDEO}")
print(f"\nExample scenes: {len(example_scenes)} total")
print(f"  Copy candidates:    {len(copy)}")
print(f"  Re-encode candidates: {len(reencode)}")

if copy:
    print("\nCopy candidates (lossless):")
    for c in copy:
        print(f"  scene {c['scene_id']}: start={c['orig_start']:.2f}s "
              f"-> snapped to {c['start']:.2f}s (diff={c['start_diff_sec']:.3f}s)")

if reencode:
    print("\nRe-encode candidates (need smartcut or re-encode):")
    for c in reencode:
        print(f"  scene {c['scene_id']}: start={c['orig_start']:.2f}s "
              f"(nearest kf at {c['start']:.2f}s, diff={c['start_diff_sec']:.3f}s)")
