import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from amverge.core.dedup import run_dedup, DEDUP_METHODS, auto_detect_method

video = sys.argv[1] if len(sys.argv) > 1 else "episode.mp4"
method = sys.argv[2] if len(sys.argv) > 2 else auto_detect_method()

if method not in DEDUP_METHODS:
    print(f"Unknown method: {method}")
    print(f"Available: {', '.join(DEDUP_METHODS.keys())}")
    sys.exit(1)

print(f"Method: {DEDUP_METHODS[method]['name']}")
print(f"Input:  {video}")

output = f"{Path(video).stem}_deduped_advanced.mp4"
print(f"Output: {output}")

if method == "ffmpeg":
    dry = False
    export_csv = None
else:
    dry = True
    export_csv = f"{Path(video).stem}_frames.csv"

if dry:
    print("Dry-run mode: analyzing without encoding...")

_, stats = run_dedup(
    video_path=video,
    output_path=output,
    method=method,
    dry_run=dry,
    export_frames=export_csv,
    progress_cb=lambda p, m: print(f"  {p}% {m}", end="\r"),
)
print()

if stats:
    print(f"  {stats['frames_in']} -> {stats['frames_out']} ({stats['pct_removed']}% removed)")
    if "cadence" in stats and stats["cadence"]:
        print(f"  Animation cadence: every {stats['cadence']} frames (confidence {stats['confidence']})")

if dry:
    print(f"\nDry run complete. Frame CSV: {export_csv}")
    print(f"To encode with these settings, set dry_run=False and call again.")
else:
    print(f"Saved: {output}")
