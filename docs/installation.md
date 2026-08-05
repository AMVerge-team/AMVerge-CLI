# Installation

## Requirements

- Python 3.11+
- `ffmpeg` and `ffprobe` on your PATH (or dropped in the working directory)

---

## Quick Install

```bash
pip install amverge
```

Covers `detect` (keyframe method), `export`, `merge`, and `info`.

---

## Extras

### TransNetV2 ML Detection

```bash
pip install amverge[ml]
```

Adds PyTorch + TransNetV2 scene detection (GPU auto-detected, CPU fallback).

```bash
amverge detect episode.mkv --method transnetv2
```

### Edge Detection

```bash
pip install amverge[edge]
```

Adds OpenCV for Canny edge-based cut detection.

```bash
amverge detect episode.mkv --method edge
```

### Discord Rich Presence

```bash
pip install amverge[discord]
```

Adds pypresence for Discord RPC status updates during long operations.

### AI Upscaling

```bash
pip install amverge[upscale]        # NVIDIA or CPU
pip install amverge[upscale-amd]    # AMD or Intel (Windows)
```

Adds torch + opencv + spandrel for AI video upscaling (ShuffleCUGAN / ArtCNN ONNX).
Anime4K shader-based upscaling uses FFmpeg only - no extra deps needed.

The method is picked from the model key, there is no `--method` flag.
Run `amverge upscale --list-models` to see every key.

```bash
amverge upscale episode.mp4 -m adore -s 2        # ml, spandrel
amverge upscale episode.mp4 -m C4F32             # onnx, ArtCNN
amverge upscale episode.mp4 -m anime4k --mode medium
```

### AI Frame Interpolation (Python RIFE)

```bash
pip install amverge[interpolation]
```

Adds torch + opencv for RIFE PyTorch CUDA/CPU frame interpolation.

PyTorch reaches the GPU through CUDA, so this path is NVIDIA only. On AMD or
Intel it runs on CPU. Use `amverge flowframes` instead, which runs RIFE on any
GPU through Vulkan. See [AMD and Intel GPUs](#amd-and-intel-gpus).

```bash
amverge interpolate episode.mp4 -f 2 -m rife4.25
amverge interpolate episode.mp4 -f 4 -m rife4.25-heavy
```

## AMD and Intel GPUs

PyTorch has no AMD backend on Windows: ROCm is Linux-only. So the PyTorch paths
(`amverge interpolate`, `-m adore` and the other ml upscale models, TransNetV2,
depth) fall back to CPU on an AMD or Intel card.

Everything else reaches the GPU. Check what your machine can do:

```bash
amverge gpu
```

The `GPU Backends` table names the card and reports each backend as active or
absent, with the reason.

| Instead of | Use | Reaches the GPU via |
|---|---|---|
| `amverge upscale -m adore` (ml) | `amverge upscale -m anime4k` | Vulkan, libplacebo |
| `amverge upscale -m adore` (ml) | `amverge upscale -m C4F32` | DirectML, needs `[upscale-amd]` |
| `amverge interpolate` | `amverge flowframes` | Vulkan, ncnn |

### ONNX upscaling on AMD

```bash
pip uninstall onnxruntime
pip install amverge[upscale-amd]
```

`[upscale-amd]` replaces `[upscale]`. Both ship a module named `onnxruntime`
and clobber each other, so uninstall the CPU one first. Windows only:
onnxruntime-directml publishes no Linux or macOS wheels.

Confirm the provider is live:

```bash
amverge gpu    # ONNX (DirectML) -> active
```

### Interpolation on AMD

`amverge flowframes` defaults to `--ai RifeNcnn`, which is Vulkan based and
runs on any vendor. The `RifeCuda`, `FlavrCuda` and `XvfiCuda` engines are
NVIDIA only.

```bash
amverge flowframes episode.mp4 -f 2 --ai RifeNcnn
```

If it exits immediately, the usual causes are a missing Vulkan runtime (update
your GPU driver) or a missing Visual C++ Redistributable.

### Anime4K on AMD

No extra install. It runs as a GLSL shader chain through FFmpeg libplacebo,
which is Vulkan based and vendor-neutral. `amverge gpu` reports
`Vulkan (libplacebo)` as active when your FFmpeg build includes it. If it does
not, Anime4K silently falls back to lanczos, which is not GPU accelerated.

### Deadframe Removal

```bash
pip install amverge[deadframes]
```

Adds OpenCV for dead (static subject) frame detection and removal.

```bash
amverge deadframes episode.mp4
amverge deadframes episode.mp4 --auto --safe
```

### Pipeline (Operation Chaining)

No extra deps beyond what each operation needs. Install at least 2 of deadframes, upscale, interpolation:

```bash
pip install amverge[deadframes,upscale,interpolation]
```

Chains operations in sequence with interactive arrow-key config:

```bash
amverge pipeline                          # interactive mode
amverge pipeline --load my-preset          # saved preset
```

### Flowframes (External)

```bash
pip install amverge[flowframes]
```

No extra Python deps. Requires Flowframes 1.42.0 Patreon installed separately.
Free 1.36.0 support planned.

```bash
amverge flowframes episode.mp4 -f 2
amverge flowframes-path "C:\Flowframes\Flowframes.exe"
```

### All at once

```bash
pip install amverge[ml,edge,discord,upscale,interpolation]
```

---

## FFmpeg

AMVerge CLI looks for `ffmpeg` / `ffprobe` in this order:

1. System PATH
2. A `bin/` folder in the current working directory

If neither is found, commands that require FFmpeg will fail with a clear error.

Download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html) or install via your package manager:

```bash
# Windows (winget)
winget install ffmpeg

# macOS
brew install ffmpeg

# Linux
sudo apt install ffmpeg
```

---

## Verify Install

```bash
amverge doctor    # full health check - shows what is installed and working
amverge gpu       # PyTorch + CUDA + GPU info
amverge version   # all dependency versions
```

---

## Development Install

```bash
git clone https://github.com/AMVerge-team/AMVerge-CLI
cd AMVerge-CLI
pip install -e .
pip install -e ".[ml,edge,discord,upscale,interpolation]"
```
