import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from amverge import upscale_model, ANIME4K_MODE_PRESETS, UPSCALE_REGISTRY

video_path = sys.argv[1] if len(sys.argv) > 1 else "episode.mp4"
output_path = "upscaled_anime4k.mp4"

if not Path(video_path).exists():
    print(f"Video not found: {video_path}")
    sys.exit(1)

entry = UPSCALE_REGISTRY["anime4k"]
print(f"Method: {entry['name']} ({entry['description']})")
print(f"Credit: {entry['credit']}")
print(f"Scale: 2x | Modes: {entry.get('modes', ['medium'])}")
print()

start_time = time.time()

def progress_cb(pct, msg):
    elapsed = time.time() - start_time
    if pct > 0 and pct < 100:
        eta = (elapsed / pct) * (100 - pct)
        print(f"\r[{pct:3d}%] {msg:40s}  ETA: {eta:5.1f}s", end="")
    else:
        print(f"\r[{pct:3d}%] {msg:40s}", end="")

upscale_model(
    "anime4k",
    video_path,
    output_path,
    scale=2,
    mode="medium",
    preset="high",
    progress_cb=progress_cb,
)

elapsed = time.time() - start_time
print(f"\nDone in {elapsed:.1f}s: {output_path}")
