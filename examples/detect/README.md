<p align="center">
  <img src="../../assets/AMVerge-CLI.gif" alt="AMVerge CLI" width="1440"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue?style=flat-square" alt="Python"/>
  <img src="https://img.shields.io/badge/pypi-amverge-22c55e?style=flat-square" alt="PyPI"/>
  <img src="https://img.shields.io/badge/license-GPL--3.0-22c55e?style=flat-square" alt="License"/>
</p>

# Scene Detection Examples

**Split a video into scenes at cut boundaries.**  
Use keyframe analysis (fast), Canny edge detection (accurate), or TransNetV2 ML (best).

---

## How It Works

```txt
video file
     ↓
detection method (keyframe / edge / TransNetV2)
     ↓
cut timestamps
     ↓
ffmpeg segment or smart cut
     ↓
.mp4 clips + .jpg thumbnails + scenes.json
```

Detection extracts scene boundaries, then materializes previews by method.
Keyframe detection uses fast stream copy because its boundaries are keyframes;
TransNetV2 re-encodes at the detected boundary so previews are frame-accurate.

---

## Examples

| File | Method | Dependencies |
|---|---|---|
| [01_basic_detect.py](01_basic_detect.py) | keyframe (default) | none |
| [02_transnetv2_detect.py](02_transnetv2_detect.py) | TransNetV2 ML | `pip install amverge[ml]` |
| [03_edge_detect.py](03_edge_detect.py) | Canny edge + cosine | `pip install amverge[edge]` |
| [04_custom_settings.py](04_custom_settings.py) | keyframe with custom params | none |

---

## Quick Start

```bash
pip install amverge

# Basic keyframe detection
python examples/detect/01_basic_detect.py episode.mp4

# ML detection (best accuracy)
pip install amverge[ml]
python examples/detect/02_transnetv2_detect.py episode.mp4

# Edge detection
pip install amverge[edge]
python examples/detect/03_edge_detect.py episode.mp4
```

---

## See Also

| | |
|---|---|
| [Library API](../../docs/library.md) | `detect_scenes()` reference |
| [Detection Methods](../../docs/detection-methods.md) | method comparison |
| [CLI Reference](../../docs/cli-reference.md) | `amverge detect` command |
