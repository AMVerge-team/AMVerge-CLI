"""Video stream metadata.

Uses the AmvergeVideo class for clean property access.

Usage:
    python 01_video_info.py [video_path]
"""

import sys
from amverge import AmvergeVideo

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "episode.mp4"

video = AmvergeVideo(VIDEO)

dur = video.duration
h, m, s = int(dur // 3600), int((dur % 3600) // 60), dur % 60
dur_str = f"{h}h {m:02d}m {s:05.2f}s" if h else f"{m}m {s:05.2f}s" if m else f"{s:.2f}s"

print(f"\n{video.name}  {dur_str}\n")
print(f" Video")
print(f"  Codec:      {video.codec}")
print(f"  Resolution: {video.width}x{video.height}")
print(f"  FPS:        {video.fps}")
print(f"  Frames:     {video.total_frames}")
print(f"  HEVC:       {video.is_hevc}")
print()

for s in video.audio_streams:
    print(f" Audio")
    print(f"  Codec:       {s['codec']}")
    print(f"  Sample rate: {s['sample_rate']} Hz")
    print(f"  Channels:    {s['channels']}")
    bps = s.get("bit_rate")
    if bps:
        print(f"  Bitrate:     {bps/1_000:.0f} kbps")

print(f"\n{len(video.keyframes)} keyframes")
print(f"  First: {', '.join(f'{t:.2f}s' for t in video.keyframes[:5])}")
print(f"  Last:  {video.keyframes[-1]:.2f}s")
