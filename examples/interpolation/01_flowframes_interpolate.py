import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from amverge import (
    flowframes_available, run_flowframes, cancel_flowframes,
    get_flowframes_path, FLOWFRAMES_VERSION,
)

video_path = sys.argv[1] if len(sys.argv) > 1 else "episode.mp4"

if not Path(video_path).exists():
    print(f"Video not found: {video_path}")
    sys.exit(1)

if not flowframes_available():
    print("Flowframes.exe (1.42.0) not found. Free 1.36.0 support planned.")
    print("  Set path:  amverge flowframes-path PATH")
    sys.exit(1)

ff_exe = get_flowframes_path()
print(f"Flowframes {FLOWFRAMES_VERSION}: {ff_exe}")
print(f"Input:  {video_path}")
print()

def progress_cb(pct, msg):
    print(f"\r[{pct:3d}%] {msg}", end="")

try:
    output = run_flowframes(
        input_path=video_path,
        output_dir=".",
        factor=2,
        ai="RifeNcnn",
        model="RIFE 4.26",
        output_format="Mp4",
        encoder="X264",
        pix_fmt="Yuv420P",
        progress_cb=progress_cb,
    )
    print()
    print(f"Done: {output}")

except KeyboardInterrupt:
    print("\nCancelling...")
    cancel_flowframes()
    sys.exit(1)

except Exception as e:
    print(f"\nError: {e}")
    sys.exit(1)
