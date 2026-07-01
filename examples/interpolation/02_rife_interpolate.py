import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from amverge.core.interpolation import (
    interpolate_video,
    INTERPOLATION_REGISTRY,
    INTERPOLATION_AVAILABLE,
    is_weight_downloaded,
    download_weights,
)
from amverge.core.infra.diagnostics import get_gpu_info

video = sys.argv[1] if len(sys.argv) > 1 else "episode.mp4"

model = "rife4.25"
factor = 2
i = 2
while i < len(sys.argv):
    if sys.argv[i] == "--model" and i + 1 < len(sys.argv):
        model = sys.argv[i + 1]
        i += 2
    elif sys.argv[i] == "--factor" and i + 1 < len(sys.argv):
        factor = int(sys.argv[i + 1])
        i += 2
    else:
        i += 1

if not INTERPOLATION_AVAILABLE:
    print("Interpolation requires torch and opencv. Run: pip install amverge[ml]")
    sys.exit(1)

gpu_info = get_gpu_info()
if gpu_info.get("cuda_available"):
    print(f"GPU: {gpu_info['gpu_name']} ({gpu_info['vram_gb']:.1f} GB VRAM)")
else:
    print("No NVIDIA GPU detected. Interpolation on CPU will be very slow.")

entry = INTERPOLATION_REGISTRY.get(model)
if entry is None:
    print(f"Unknown model: {model}")
    print(f"Available: {', '.join(INTERPOLATION_REGISTRY.keys())}")
    sys.exit(1)

print(f"Model: {entry['name']}  Factor: {factor}x")
print(f"Input: {video}")

if not is_weight_downloaded(model):
    print(f"Downloading {entry['name']}...")
    download_weights(model, progress_cb=lambda p, m: print(f"  {p}% {m}", end="\r"))
    print()

output = f"{Path(video).stem}_{factor}x_{model}.mp4"
print(f"Output: {output}")

print("Interpolating...")
interpolate_video(
    input_path=video,
    output_path=output,
    model_key=model,
    factor=factor,
    preset="high",
    progress_cb=lambda p, m: print(f"  {p}% {m}", end="\r"),
)
print(f"\nSaved: {output}")
