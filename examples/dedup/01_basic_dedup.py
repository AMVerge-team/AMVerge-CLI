import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from amverge.core.dedup import (
    run_dedup_simple,
    auto_detect_method,
    PRESETS,
)

video = sys.argv[1] if len(sys.argv) > 1 else "episode.mp4"
preset = sys.argv[2] if len(sys.argv) > 2 else "normal"

method = auto_detect_method()
preset_keys = ["aggressive", "anime", "normal", "gentle"]
if preset not in preset_keys:
    print(f"Unknown preset: {preset}. Choose from: {', '.join(preset_keys)}")
    sys.exit(1)

print(f"Method: {method} (auto-detected)")
print(f"Preset: {preset}")
print(f"Input:  {video}")

output = f"{Path(video).stem}_deduped.mp4"
print(f"Output: {output}")

print("Processing...")
_, stats = run_dedup_simple(
    video_path=video,
    output_path=output,
    preset=preset,
    progress_cb=lambda p, m: print(f"  {p}% {m}", end="\r"),
)
print()
print(f"  {stats['frames_in']} -> {stats['frames_out']} ({stats['pct_removed']}% removed)")
if "cadence" in stats and stats["cadence"]:
    print(f"  Animation cadence: every {stats['cadence']} frames (confidence {stats['confidence']})")
print(f"Saved: {output}")
