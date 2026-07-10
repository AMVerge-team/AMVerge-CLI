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
pip install amverge[upscale]
```

Adds torch + opencv + spandrel for AI video upscaling (ShuffleCUGAN / ArtCNN ONNX).
Anime4K shader-based upscaling uses FFmpeg only - no extra deps needed.

```bash
amverge upscale episode.mp4 --method ml --model adore -s 2
amverge upscale episode.mp4 --method artcnn --artcnn-model C4F32
amverge upscale episode.mp4 --method anime4k --anime4k-mode medium
```

### AI Frame Interpolation (Python RIFE)

```bash
pip install amverge[interpolation]
```

Adds torch + opencv for RIFE PyTorch CUDA/CPU frame interpolation.

```bash
amverge interpolate episode.mp4 -f 2 -m rife4.25
amverge interpolate episode.mp4 -f 4 -m rife4.25-heavy
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
