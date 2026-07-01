# Interpolation Examples

Frame interpolation via Python RIFE (PyTorch) and Flowframes 1.42.0.

Requirements:

- For Python RIFE: `pip install amverge[ml]` (PyTorch + OpenCV)
- For Flowframes: Flowframes 1.42.0 Patreon installed, NVIDIA GPU recommended
- Set path: `amverge flowframes-path PATH`

## Examples

| File | What It Does |
|------|-------------|
| `01_flowframes_interpolate.py` | Run Flowframes 1.42.0 with RIFE NCNN, 2x factor |
| `02_rife_interpolate.py` | Run Python RIFE inference (PyTorch CUDA/CPU) |

### Python RIFE

```bash
python examples/interpolation/02_rife_interpolate.py episode.mp4
python examples/interpolation/02_rife_interpolate.py episode.mp4 --model rife4.25-heavy --factor 4
```

Requires `pip install amverge[ml]`. CUDA auto-detected, CPU fallback.

### Flowframes

```bash
python examples/interpolation/01_flowframes_interpolate.py episode.mp4
```
