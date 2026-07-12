<p align="center">
  <img src="assets/AMVerge-CLI.gif" alt="AMVerge CLI" width="1440"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue?style=flat-square" alt="Python"/>
  <a href="https://pypi.org/project/amverge/"><img src="https://img.shields.io/badge/pypi-amverge-22c55e?style=flat-square" alt="PyPI"/></a>
  <img src="https://img.shields.io/badge/license-GPL--3.0-22c55e?style=flat-square" alt="License"/>
</p>

# AMVerge CLI — Depth Map

Per-frame monocular depth estimation via [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) by Lihe Yang et al. (NeurIPS 2024).

---

## Install

```bash
pip install amverge[depth]
```

---

## Quick Start

```bash
amverge depth-map episode.mp4                              # side-by-side original + depth
amverge depth-map episode.mp4 --pred-only                  # depth map only
amverge depth-map episode.mp4 --pred-only --grayscale      # grayscale depth
amverge depth-map episode.mp4 -e vitl --input-size 700     # large model, high detail
amverge depth-map episode.mp4 -c turbo                     # turbo colormap
```

```python
from amverge import generate_depth_map

generate_depth_map("episode.mp4", "depth.mp4", encoder="vits",
                   pred_only=True, grayscale=True)
```

---

## Models

| Encoder | Params | Size |
|---------|--------|------|
| vits (Small) | 24.8M | ~95 MB |
| vitb (Base) | 97.5M | ~372 MB |
| vitl (Large) | 335.3M | ~1.3 GB |

Auto-downloaded on first run from [AniSmooth-Models](https://github.com/AniScripts/AniSmooth-Models/releases/tag/depth). Stored at `%APPDATA%/com.amverge.cli/models/depth/`.

Manage with: `amverge models --depth`, `amverge models --download vits`, `amverge models --delete vits`.

---

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--encoder`, `-e` | vits | Model size: vits, vitb, vitl |
| `--input-size`, `-s` | 518 | Inference resolution (higher = more detail) |
| `--colormap`, `-c` | inferno | Color palette: inferno, viridis, plasma, magma, turbo, jet, twilight, hot, cool, rainbow, ocean, bone, winter, summer |
| `--pred-only` | off | Output depth map only (no original video) |
| `--grayscale` | off | Grayscale depth (no color palette) |
| `--no-monitor` | off | Disable GPU/CPU/RAM monitoring |
| `--yes`, `-y` | off | Auto-confirm model download |

---

## How It Works

```txt
depth_anything_v2.dpt.DepthAnythingV2  (PyTorch, GPU/CPU)
        ↓ per-frame inference
depth map → normalize → colormap / grayscale
        ↓
FFmpeg pipe (rawvideo → libx264 + AAC)  →  output.mp4
```

Output is always H.264 + AAC in MP4 container with `+faststart`. Source audio is muxed into the output automatically.

---

## Examples

```bash
python examples/depth/01_basic_depth_map.py episode.mp4
python examples/depth/02_advanced_depth_map.py episode.mp4 vitb
```

See [examples/depth/](examples/depth/) for more.

---

## Files

```
amverge/core/depth/
├── __init__.py          exports: generate_depth_map, DEPTH_AVAILABLE, MODEL_CONFIGS, COLMAPS
└── depth_map.py         model download, per-frame inference, FFmpeg pipe, audio mux

amverge/commands/depth/
└── depth_map.py         CLI command with Rich progress + SystemMonitor

examples/depth/
├── 01_basic_depth_map.py
├── 02_advanced_depth_map.py
└── README.md
```

---

## Credits

- Depth-Anything-V2 by [Lihe Yang et al.](https://github.com/DepthAnything/Depth-Anything-V2) (NeurIPS 2024)
- Model weights hosted at [AniSmooth-Models](https://github.com/AniScripts/AniSmooth-Models)
