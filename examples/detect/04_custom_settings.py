"""Custom detection settings - fine-tune the keyframe method.

Shows how to control min scene duration, workers, similarity threshold,
and disable thumbnails or similarity checks for faster runs.

Usage:
    python 04_custom_settings.py [video_path]
"""

import sys
from pathlib import Path
from amverge import detect_scenes

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "episode.mp4"

result = detect_scenes(
    video_path=VIDEO,
    method="keyframe",
    min_duration=1.0,              # drop scenes shorter than 1 second
    thumbnails=False,              # skip thumbnails (faster)
    similarity=False,              # skip similarity check
    progress=lambda stage, pct, msg: print(f"  [{stage}] {pct}% {msg}"),
)

print(f"\n{len(result.scenes)} scenes detected (min 1.0s, no thumbnails)")
for scene in result.scenes:
    print(f"  [{scene.index:02d}] {scene.start:6.2f}s - {scene.end:6.2f}s  "
          f"({scene.duration:5.2f}s)")

# For even more control, use the low-level API:
from amverge import detect_cuts_by_keyframe, run_ffmpeg_segment, collect_scenes
from amverge import get_video_duration

cuts = detect_cuts_by_keyframe(VIDEO, min_duration=1.0)
print(f"\nLow-level API: {len(cuts)} cut points")
print(f"  Cuts at: {', '.join(f'{c:.2f}s' for c in cuts[:5])}{'...' if len(cuts) > 5 else ''}")

dur = get_video_duration(VIDEO)
output_pattern = str(Path(VIDEO).parent / f"{Path(VIDEO).stem}_custom" / f"{Path(VIDEO).stem}_%04d.mp4")
run_ffmpeg_segment(VIDEO, output_pattern, cuts)
scenes = collect_scenes(str(Path(output_pattern).parent), Path(VIDEO).stem, cuts, dur)

print(f"  Low-level output: {len(scenes)} clips written")
