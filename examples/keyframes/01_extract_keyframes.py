"""Keyframe extraction - V1 and V2 methods.

V1: generate_keyframes (PyAV packet demux with progress callback)
V2: get_keyframe_timestamps_pyav (PyAV demux, Discard.nonkey enum)

Usage:
    python 01_extract_keyframes.py [video_path]
"""

import sys
from amverge import get_keyframe_timestamps_pyav, generate_keyframes

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "episode.mp4"

# V2 method (used by TransNetV2 pipeline)
kf_v2 = get_keyframe_timestamps_pyav(VIDEO)
print(f"V2: {len(kf_v2)} keyframes")
print(f"    First 5: {', '.join(f'{t:.2f}s' for t in kf_v2[:5])}")
if len(kf_v2) > 5:
    print(f"    Last:    {kf_v2[-1]:.2f}s")

# V1 method (with progress callback)
print()
kf_v1 = generate_keyframes(
    VIDEO,
    progress_cb=lambda pct, msg: print(f"  V1 progress: {pct}% {msg}"),
)
print(f"\nV1: {len(kf_v1)} keyframes (same count: {len(kf_v1) == len(kf_v2)})")

# Stats
gaps = [kf_v2[i+1] - kf_v2[i] for i in range(len(kf_v2) - 1)]
if gaps:
    print(f"\nGap stats: min={min(gaps):.2f}s  max={max(gaps):.2f}s  "
          f"avg={sum(gaps)/len(gaps):.2f}s")
