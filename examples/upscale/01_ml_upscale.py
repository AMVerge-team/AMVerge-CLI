import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from amverge import (
    upscale_model, SystemMonitor, format_eta,
    QUALITY_PRESETS, UPSCALE_AVAILABLE, UPSCALE_REGISTRY,
)

video_path = sys.argv[1] if len(sys.argv) > 1 else "episode.mp4"
output_path = "upscaled_ml.mp4"

if not Path(video_path).exists():
    print(f"Video not found: {video_path}")
    sys.exit(1)

if not UPSCALE_AVAILABLE:
    print("ML upscaling not available. Run: pip install amverge[upscale]")
    sys.exit(1)

model_key = "adore"
entry = UPSCALE_REGISTRY[model_key]
print(f"Model: {entry['name']} ({entry['description']})")
print(f"Credit: {entry['credit']}")
print(f"Scales: {entry['scales']}")
print(f"Preset: high (CRF {QUALITY_PRESETS['high']['crf']}, {QUALITY_PRESETS['high']['x264']})")
print()

monitor = SystemMonitor(enabled=True)

def progress_cb(pct, msg):
    monitor.progress_callback(pct, msg)
    s = monitor.stats
    line = f"\r[{pct:3d}%] {s['elapsed_s']:5.1f}s"
    if s.get("eta_s"):
        line += f"  ETA: {format_eta(s['eta_s'])}"
    if s.get("gpu_util") is not None:
        line += f"  GPU: {s['gpu_util']:.0f}%"
    if s.get("vram_used_mb") is not None:
        line += f"  VRAM: {s['vram_used_mb']:.0f} MB"
    if s.get("cpu_percent") is not None:
        line += f"  CPU: {s['cpu_percent']:.0f}%"
    print(line, end="")

upscale_model(
    model_key,
    video_path,
    output_path,
    scale=2,
    preset="high",
    progress_cb=progress_cb,
    monitor=monitor,
)

s = monitor.stats
print(f"\nDone in {s['elapsed_s']:.1f}s: {output_path}")
