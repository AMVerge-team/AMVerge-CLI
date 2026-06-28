"""Probe diagnostics - shown by `amverge probe`.

Uses ffprobe for metadata and PyAV for keyframe extraction.

Usage:
    python 02_probe.py [video_path]
"""

import sys
from pathlib import Path
from amverge import (
    probe_video_fps, probe_video_dimensions, probe_video_duration,
    probe_video_total_frames, check_if_hevc,
    get_keyframe_timestamps_pyav, build_video_cache_prefix,
)

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "episode.mp4"

fps = probe_video_fps(VIDEO)
w, h = probe_video_dimensions(VIDEO)
dur = probe_video_duration(VIDEO)
frames = probe_video_total_frames(VIDEO, fps, dur)

print(f"\n{Path(VIDEO).name}  {dur:.2f}s\n")
print(" Video")
print(f"  Duration:   {dur:.2f}s")
print(f"  Resolution: {w}x{h}")
print(f"  FPS:        {fps:.3f}")
print(f"  Frames:     ~{frames}")
print(f"  Codec:      {'HEVC' if check_if_hevc(VIDEO) else 'H.264 / other'}")
print()

kf = get_keyframe_timestamps_pyav(VIDEO)
gaps = [kf[i+1] - kf[i] for i in range(len(kf) - 1)]
avg_gap = sum(gaps) / len(gaps) if gaps else 0

print(" Keyframes")
print(f"  Count:      {len(kf)}")
print(f"  Avg. gap:   {avg_gap:.2f}s")

prefix = build_video_cache_prefix(Path(VIDEO))
secs_path = Path(VIDEO).parent / f"{prefix}_secs.npy"
print()
print(" Scene Cache")
print(f"  Prefix:     {prefix}")
print(f"  Location:   {Path(VIDEO).parent}")
print(f"  Status:     {'cached' if secs_path.exists() else 'no cache - TransNetV2 will run on next detect'}")
