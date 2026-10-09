<p align="center">
  <img src="../../assets/AMVerge-CLI.gif" alt="AMVerge CLI" width="1440"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue?style=flat-square" alt="Python"/>
  <img src="https://img.shields.io/badge/pypi-amverge-22c55e?style=flat-square" alt="PyPI"/>
  <img src="https://img.shields.io/badge/license-GPL--3.0-22c55e?style=flat-square" alt="License"/>
</p>

# Keyframe Examples

**Extract I-frame timestamps and snap scene boundaries to them.**  
A "copy" mode cut is always widened outward to the nearest enclosing
keyframes - never trimmed, never encoded, just a plain stream copy.

---

## How It Works

```txt
video file
     ↓
PyAV packet demux (every packet, filtered by is_keyframe)
     ↓
sorted keyframe timestamps (seconds)
     ↓
snap_range_to_keyframes(start, end)
     ↓
[preceding keyframe, following keyframe) - the actual copy range
```

The V1 method (`generate_keyframes`) supports progress callbacks.
The V2 method (`get_keyframe_timestamps_pyav`) never sets
`stream.discard` - doing so corrupts PTS on B-frame sources.

---

## Examples

| File | Description |
|---|---|
| [01_extract_keyframes.py](01_extract_keyframes.py) | V1 + V2 keyframe extraction with stats |
| [02_align_scenes.py](02_align_scenes.py) | snap scene boundaries outward to keyframes |

---

## Quick Start

```bash
pip install amverge

# Extract keyframes
python examples/keyframes/01_extract_keyframes.py episode.mp4

# Snap sample scenes to keyframes
python examples/keyframes/02_align_scenes.py episode.mp4
```

---

## See Also

| | |
|---|---|
| [Library API](../../docs/library.md) | `get_keyframe_timestamps_pyav()`, `snap_range_to_keyframes()` |
| [Detection Methods](../../docs/detection-methods.md) | how keyframes drive the cut pipeline |
