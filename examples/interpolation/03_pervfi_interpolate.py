import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from amverge.core.interpolation import (
    interpolate_video,
    INTERPOLATION_REGISTRY,
    INTERPOLATION_AVAILABLE,
    get_pervfi_models,
    is_weight_downloaded,
    download_weights,
)
from amverge.core.infra.diagnostics import get_gpu_info

video = sys.argv[1] if len(sys.argv) > 1 else "episode.mp4"

model = "pervfi"
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
    print("Interpolation requires torch and opencv. Run: pip install amverge[interpolation]")
    sys.exit(1)

gpu_info = get_gpu_info()
if gpu_info.get("cuda_available"):
    print(f"GPU: {gpu_info['gpu_name']} ({gpu_info['vram_gb']:.1f} GB VRAM)")
else:
    print("No NVIDIA GPU detected. PerVFI on CPU will be very slow.")

pervfi_models = get_pervfi_models()
if model not in pervfi_models:
    print(f"Unknown PerVFI model: {model}")
    print(f"Available: {', '.join(pervfi_models.keys())}")
    sys.exit(1)

entry = pervfi_models[model]
print(f"Model: {entry['name']}  Factor: {factor}x")
print(f"Description: {entry['description']}")
print(f"Flow estimator: {entry['flow_estimator']}")
print(f"Input: {video}")

if not is_weight_downloaded(model):
    print(f"Downloading {entry['name']} generator weights...")
    download_weights(model, progress_cb=lambda p, m: print(f"  {p}% {m}", end="\r"))
    print()
    print("GMFlow flow estimator weights will download on first use.")

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
