# Interpolation Examples

Frame interpolation via Python RIFE (PyTorch), PerVFI (GMFlow + PerVFI generator), and Flowframes 1.42.0 (free 1.36.0 planned).

Requirements:

- For Python RIFE / PerVFI: `pip install amverge[interpolation]` (PyTorch + OpenCV + scipy)
- For Flowframes: Flowframes 1.42.0 Patreon installed, NVIDIA GPU recommended
  - Support for free Flowframes 1.36.0 is planned (delivery TBD since it differs from 1.42.0)
- Set path: `amverge flowframes-path PATH`

## Examples

| File | What It Does |
|------|-------------|
| `01_flowframes_interpolate.py` | Run Flowframes 1.42.0 with RIFE NCNN, 2x factor |
| `02_rife_interpolate.py` | Run Python RIFE inference (PyTorch CUDA/CPU) |
| `03_pervfi_interpolate.py` | Run PerVFI inference (GMFlow flow estimator + PerVFI generator, PyTorch CUDA/CPU) |

### Python RIFE

```bash
python examples/interpolation/02_rife_interpolate.py episode.mp4
python examples/interpolation/02_rife_interpolate.py episode.mp4 --model rife4.25-heavy --factor 4
```

Requires `pip install amverge[interpolation]`. CUDA auto-detected, CPU fallback.

### PerVFI

```bash
python examples/interpolation/03_pervfi_interpolate.py episode.mp4
python examples/interpolation/03_pervfi_interpolate.py episode.mp4 --model pervfi-vb --factor 2
```

PerVFI uses a two-stage pipeline: GMFlow optical flow estimator (auto-downloads ~300 MB checkpoint) + PerVFI generator (auto-downloads ~35 MB checkpoint). The `pervfi` model uses a normalizing flow decoder for best quality; `pervfi-vb` uses a faster multi-scale decoder.

Requires `pip install amverge[interpolation]`. GPU strongly recommended (12 GB VRAM for 1080p).

### Flowframes

```bash
python examples/interpolation/01_flowframes_interpolate.py episode.mp4
```
