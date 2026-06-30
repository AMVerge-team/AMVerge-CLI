import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from amverge import upscale_model, ANIME4K_MODE_PRESETS

video_path = sys.argv[1] if len(sys.argv) > 1 else "episode.mp4"
output_path = "upscaled_anime4k.mp4"

if not Path(video_path).exists():
    print(f"Video not found: {video_path}")
    sys.exit(1)

print(f"Upscaling: {video_path} -> {output_path}")
print(f"Model: anime4k (FFmpeg lanczos+unsharp+smartblur)")
print(f"Mode: medium, Scale: 2x")
print(f"Available modes: {list(ANIME4K_MODE_PRESETS.keys())}")
print(f"  light  - fast, minimal filtering")
print(f"  medium - balanced quality/speed")
print(f"  strong - maximum enhancement")

upscale_model(
    "anime4k",
    video_path,
    output_path,
    scale=2,
    mode="medium",
    preset="high",
    progress_cb=lambda pct, msg: print(f"[{pct}%] {msg}"),
)

print(f"Done: {output_path}")
