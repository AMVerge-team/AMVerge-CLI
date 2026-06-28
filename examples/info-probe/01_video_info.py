"""Video stream metadata - shown by `amverge info`.

Uses PyAV to read container metadata without decoding frames.

Usage:
    python 01_video_info.py [video_path]
"""

import sys
from pathlib import Path
from amverge import get_video_info, get_video_duration

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "episode.mp4"

dur = get_video_duration(VIDEO)
h, m, s = int(dur // 3600), int((dur % 3600) // 60), dur % 60
dur_str = f"{h}h {m:02d}m {s:05.2f}s" if h else f"{m}m {s:05.2f}s" if m else f"{s:.2f}s"

info = get_video_info(VIDEO)
print(f"\n{Path(VIDEO).name}  {dur_str}\n")

for stream in info["streams"]:
    if stream["type"] == "video":
        print(" Video")
        print(f"  Codec:      {stream['codec']}")
        print(f"  Resolution: {stream['width']}x{stream['height']}")
        print(f"  FPS:        {stream['fps']}")
        bps = stream.get("bit_rate")
        if bps:
            print(f"  Bitrate:    {bps/1_000_000:.1f} Mbps")
    elif stream["type"] == "audio":
        print(" Audio")
        print(f"  Codec:       {stream['codec']}")
        print(f"  Sample rate: {stream['sample_rate']} Hz")
        print(f"  Channels:    {stream['channels']}")
        bps = stream.get("bit_rate")
        if bps:
            print(f"  Bitrate:     {bps/1_000:.0f} kbps")
