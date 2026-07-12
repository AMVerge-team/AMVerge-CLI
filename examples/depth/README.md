# Depth Map Examples

Per-frame monocular depth estimation using Depth-Anything-V2.

Requirements:

- `pip install amverge[depth]` (PyTorch + OpenCV + depth-anything-v2)
- Models auto-downloaded from GitHub Releases on first run

## Examples

| File | What It Does |
|------|-------------|
| `01_basic_depth_map.py` | Side-by-side depth map with inferno colormap (vits encoder) |
| `02_advanced_depth_map.py` | Multiple output variants (side-by-side, depth-only, grayscale) with vitb encoder |

### Basic Depth Map

```bash
python 01_basic_depth_map.py my_video.mp4
```

Outputs `my_video_depth.mp4` - side-by-side original + depth visualization.

### Advanced Variants

```bash
python 02_advanced_depth_map.py my_video.mp4 [vits|vitb|vitl]
```

Produces three output files:
- `my_video_depth_side.mp4` - side-by-side with inferno colormap
- `my_video_depth_only.mp4` - depth map only with turbo colormap
- `my_video_depth_gray.mp4` - grayscale depth map

## CLI Equivalent

```bash
amverge depth-map video.mp4                        # side-by-side (default)
amverge depth-map video.mp4 --pred-only            # depth map only
amverge depth-map video.mp4 --pred-only --grayscale  # grayscale depth
amverge depth-map video.mp4 -e vitl --input-size 700  # large model, high detail
```

## Model Sources

- Depth-Anything-V2: https://github.com/DepthAnything/Depth-Anything-V2
- Model weights: https://github.com/AniScripts/AniSmooth-Models/releases/tag/depth
- Credit: Lihe Yang et al. (NeurIPS 2024)
