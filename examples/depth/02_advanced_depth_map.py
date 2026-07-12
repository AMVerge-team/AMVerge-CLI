import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from amverge.core.depth import (
    generate_depth_map,
    DEPTH_AVAILABLE,
    MODEL_CONFIGS,
    COLMAPS,
    is_model_downloaded,
    download_model,
)
from amverge.core.infra.diagnostics import get_gpu_info

video = sys.argv[1] if len(sys.argv) > 1 else "episode.mp4"
encoder = "vitb" if len(sys.argv) < 3 else sys.argv[2]

if not DEPTH_AVAILABLE:
    print("Depth-Anything-V2 not installed. Run: pip install amverge[depth]")
    sys.exit(1)

if encoder not in MODEL_CONFIGS:
    print(f"Unknown encoder: {encoder}. Choose from: {list(MODEL_CONFIGS.keys())}")
    sys.exit(1)

gpu_info = get_gpu_info()
device = "cuda" if gpu_info.get("cuda_available") else "cpu"
print(f"Device: {device}  GPU: {gpu_info.get('gpu_name', 'N/A')}")

print(f"Encoder: {encoder}  Input size: 518")
print(f"Input: {video}")

if not is_model_downloaded(encoder):
    print(f"Downloading Depth-Anything-V2-{encoder} model...")
    download_model(encoder, progress_cb=lambda p, m: print(f"  {p}% {m}", end="\r"))
    print()

variants = {
    "side-by-side (inferno)": {
        "output": f"{Path(video).stem}_depth_side.mp4",
        "pred_only": False,
        "grayscale": False,
    },
    "depth-only (turbo)": {
        "output": f"{Path(video).stem}_depth_only.mp4",
        "pred_only": True,
        "grayscale": False,
    },
    "depth-only (grayscale)": {
        "output": f"{Path(video).stem}_depth_gray.mp4",
        "pred_only": True,
        "grayscale": True,
    },
}

for label, opts in variants.items():
    print(f"\nGenerating {label}...")
    kwargs = {k: v for k, v in opts.items() if k != "colormap"}
    kwargs["colormap"] = opts.get("colormap", "turbo" if opts.get("pred_only") else "inferno")
    generate_depth_map(
        input_path=video,
        output_path=opts["output"],
        encoder=encoder,
        progress_cb=lambda p, m: print(f"  {p}% {m}", end="\r"),
        **kwargs,
    )
    print(f"\n  Saved: {opts['output']}")
