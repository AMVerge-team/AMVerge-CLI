<p align="center">
  <img src="../../assets/AMVerge-CLI.gif" alt="AMVerge CLI" width="1440"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue?style=flat-square" alt="Python"/>
  <img src="https://img.shields.io/badge/pypi-amverge-22c55e?style=flat-square" alt="PyPI"/>
  <img src="https://img.shields.io/badge/license-GPL--3.0-22c55e?style=flat-square" alt="License"/>
</p>

# Scene Cutting Examples

**Cut video into clips using smart cut or FFmpeg segment muxer.**  
One mode for the whole batch: `copy` (true stream copy) or `reencode` (exact boundaries).

---

## How It Works

```txt
scene boundaries + keyframes
     ↓
mode="copy"                          mode="reencode"
     ↓                                    ↓
snap [start, end) outward             re-encode the exact
to the enclosing keyframes            [start, end) range
     ↓                                    ↓
one plain `-c copy`                   libx264/NVENC
     ↓
parallel cut via ThreadPoolExecutor
```

`copy` is always a true, unmodified stream copy - no edit lists, no
trimming, no silent fallback to re-encode. The trade-off: a clip can carry
up to one keyframe interval of the neighboring scene at either edge.
`reencode` cuts the exact boundary at the cost of a real encode.

The V1 pipeline (`run_ffmpeg_segment`) uses `ffmpeg -segment_times` with
stream copy for lossless splitting - same trade-off as `copy` mode above,
since a stream-copy split can only land on a real keyframe.

---

## Examples

| File | Description |
|---|---|
| [01_smart_cut.py](01_smart_cut.py) | V2 pipeline: `copy` or `reencode`, whole batch |
| [02_ffmpeg_segment.py](02_ffmpeg_segment.py) | V1 pipeline: FFmpeg segment muxer |
| [03_single_scene.py](03_single_scene.py) | cut one scene, inspect the keyframe snap |

---

## Quick Start

```bash
pip install amverge[ml]

# Smart cut (V2)
python examples/cutting/01_smart_cut.py episode.mp4 copy
python examples/cutting/01_smart_cut.py episode.mp4 reencode

# FFmpeg segment (V1)
python examples/cutting/02_ffmpeg_segment.py episode.mp4

# Single scene test
python examples/cutting/03_single_scene.py episode.mp4
```

---

## See Also

| | |
|---|---|
| [Library API](../../docs/library.md) | `cut_scene()`, `cut_all_scenes()`, `run_ffmpeg_segment()` |
| [Detection Methods](../../docs/detection-methods.md) | cut mode comparison table |
