import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from amverge.core.depth import (
    generate_depth_map,
    DEPTH_AVAILABLE,
    MODEL_CONFIGS,
    is_model_downloaded,
    download_model,
)
from amverge.core.infra.diagnostics import get_gpu_info

video = sys.argv[1] if len(sys.argv) > 1 else "episode.mp4"

if not DEPTH_AVAILABLE:
    print("Depth-Anything-V2 not installed. Run: pip install amverge[depth]")
    sys.exit(1)

gpu_info = get_gpu_info()
if gpu_info.get("cuda_available"):
    print(f"GPU: {gpu_info['gpu_name']} ({gpu_info['vram_gb']:.1f} GB VRAM)")
else:
    print("No NVIDIA GPU detected. Depth estimation on CPU will be slow.")

encoder = "vits"
print(f"Encoder: {encoder}")
print(f"Input: {video}")

if not is_model_downloaded(encoder):
    print(f"Downloading Depth-Anything-V2-{encoder} model...")
    download_model(encoder, progress_cb=lambda p, m: print(f"  {p}% {m}", end="\r"))
    print()

output = f"{Path(video).stem}_depth.mp4"
print(f"Output: {output}")

print("Processing...")
generate_depth_map(
    input_path=video,
    output_path=output,
    encoder=encoder,
    progress_cb=lambda p, m: print(f"  {p}% {m}", end="\r"),
)
print(f"\nSaved: {output}")
