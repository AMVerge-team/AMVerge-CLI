"""FFmpeg segment muxer - V1 pipeline cutting.

Uses ffmpeg -segment_times with stream copy. Fast, lossless.
Handles Windows 32767-char command line limit via chunking.

Usage:
    python 02_ffmpeg_segment.py [video_path]
"""

import sys
from pathlib import Path
from amverge import (
    detect_cuts_by_keyframe, run_ffmpeg_segment, collect_scenes,
    get_video_duration,
)

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "episode.mp4"
stem = Path(VIDEO).stem

cuts = detect_cuts_by_keyframe(VIDEO, min_duration=0.25)
print(f"{len(cuts)} cut points from keyframe detection")

out_dir = Path("examples_segment_output")
out_dir.mkdir(parents=True, exist_ok=True)

pattern = str(out_dir / f"{stem}_%04d.mp4")
run_ffmpeg_segment(VIDEO, pattern, cuts)

dur = get_video_duration(VIDEO)
scenes = collect_scenes(str(out_dir), stem, cuts, dur)

print(f"\n{len(scenes)} clips written to {out_dir.resolve()}")
for s in scenes[:5]:
    print(f"  scene_{s['scene_index']:04d}.mp4  {s['start']:.1f}s - {s['end']:.1f}s")
if len(scenes) > 5:
    print(f"  ... and {len(scenes) - 5} more")
