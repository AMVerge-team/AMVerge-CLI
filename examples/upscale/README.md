# Upscale Examples

AI video upscaling with three methods:

- **ML** - RealCUGAN neural network (needs `amverge[upscale]`)
- **Anime4K** - FFmpeg filter-based (lanczos+unsharp+smartblur, no ML deps)
- **ArtCNN** - Lightweight CNN via ONNX Runtime (needs `amverge[upscale]`)

## Adding Models

Models are defined in `core/upscaling/registry.json`. See [`registry.example.json`](registry.example.json) for the format.

## Requirements

```bash
pip install amverge[upscale]         # for ML and ArtCNN methods
amverge models --download adore      # pre-download ML model
amverge models --download C4F32      # pre-download ArtCNN model
amverge models --download anime4k    # pre-download Anime4K shaders
```

## Examples

| File | What It Does |
|------|-------------|
| `01_ml_upscale.py` | Upscale with adore model (2x) |
| `02_anime4k_upscale.py` | Fast filter-based upscale with Anime4K |
| `03_manage_models.py` | List, download, and delete model files |
| `registry.example.json` | Annotated registry format for adding models |
