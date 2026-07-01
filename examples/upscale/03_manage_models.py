import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from amverge import (
    UPSCALE_REGISTRY, MODEL_FILES, UPSCALE_MODEL_KEYS, ARTCNN_MODELS,
    is_weight_downloaded, download_weights, get_weight_path,
    get_ml_models, get_onnx_models, get_shader_models,
)
from amverge.core.interpolation import (
    INTERPOLATION_REGISTRY,
    is_weight_downloaded as interp_is_downloaded,
    download_weights as interp_download,
    get_weight_path as interp_weight_path,
)

print("=== Browse Upscale Registry ===")
for key, entry in UPSCALE_REGISTRY.items():
    scales = "/".join(f"{s}x" for s in entry["scales"])
    print(f"  {entry['name']:20s}  method={entry['method']:6s}  scales={scales}")
    print(f"    {entry.get('description', '')}")
    print(f"    Credit: {entry.get('credit', '')}")
    print()

print("=== Browse Interpolation Registry ===")
for key, entry in INTERPOLATION_REGISTRY.items():
    heavy_tag = " [heavy]" if entry.get("heavy") else ""
    print(f"  {entry['name']:20s}  method=rife{heavy_tag}")
    print(f"    {entry.get('description', '')}")
    print(f"    Credit: {entry.get('credit', '')}")
    print()

print("=== Query by Method ===")
print(f"  ML models:   {list(get_ml_models().keys())}")
print(f"  Shader:      {list(get_shader_models().keys())}")
print(f"  ONNX:        {list(get_onnx_models().keys())}")
print(f"  RIFE:        {list(INTERPOLATION_REGISTRY.keys())}")

print()
print("=== Downloaded Status (Upscale) ===")
for key in UPSCALE_MODEL_KEYS:
    status = "downloaded" if is_weight_downloaded(key) else "not downloaded"
    path = get_weight_path(key)
    size_mb = Path(path).stat().st_size / (1024 * 1024) if Path(path).exists() else 0
    print(f"  {key:20s}  [{status:14s}]  {size_mb:.1f} MB")

print()
print("=== Downloaded Status (Interpolation) ===")
for key in INTERPOLATION_REGISTRY:
    status = "downloaded" if interp_is_downloaded(key) else "not downloaded"
    path = interp_weight_path(key)
    size_mb = Path(path).stat().st_size / (1024 * 1024) if Path(path).exists() else 0
    print(f"  {key:20s}  [{status:14s}]  {size_mb:.1f} MB")

if len(sys.argv) > 1 and sys.argv[1] == "--download-all":
    print()
    print("Downloading all upscale models...")
    for key in UPSCALE_MODEL_KEYS:
        if not is_weight_downloaded(key):
            start = time.time()
            download_weights(key, progress_cb=lambda p, m: print(f"\r  {key}: {p}%", end=""))
            print(f"\r  {key}: downloaded in {time.time()-start:.1f}s")
        else:
            print(f"  {key}: already cached")

    print()
    print("Downloading all interpolation models...")
    for key in INTERPOLATION_REGISTRY:
        if not interp_is_downloaded(key):
            start = time.time()
            interp_download(key, progress_cb=lambda p, m: print(f"\r  {key}: {p}%", end=""))
            print(f"\r  {key}: downloaded in {time.time()-start:.1f}s")
        else:
            print(f"  {key}: already cached")
