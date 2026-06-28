# AMVerge CLI

**Scene detection and clip management — as a CLI tool and Python library.**  
Port of the AMVerge desktop app backend. Split videos into scenes, export clips, merge fragments, and build your own tools on top of it.

---

## Features

- Keyframe-based scene splitting (fast, no re-encode)
- Edge + cosine-similarity detection for difficult encodes
- Auto-generated scene thumbnails
- Duplicate / similar scene detection
- Scene export with codec selection
- Clip merging via FFmpeg concat
- Video metadata inspection
- Interactive wizard mode (`amverge` with no args)
- Usable as a Python library

---

## How It Works

```txt
amverge CLI  /  Python library
          ↓
   amverge_cli package
          ↓
    PyAV  +  FFmpeg
```

### Detection

Extracts keyframe timestamps from the video using PyAV packet demux (fast path) or decode fallback.  
Filters out scenes shorter than `min_duration`, then segments using `ffmpeg -segment_times`.

### Thumbnails

Decoded via PyAV and written as JPEG using Pillow. Generated in parallel with `ThreadPoolExecutor`.

### Similarity

Compares adjacent scene thumbnails using cosine similarity on flattened pixel arrays.  
Pairs below the threshold are flagged as potential duplicates.

---

## Repository Structure

```txt
AMVerge-CLI/
│
├── amverge_cli/
│   ├── cli.py                  entry point
│   ├── pipeline.py             high-level detect_scenes() API
│   ├── wizard.py               interactive session
│   ├── ui.py                   shared Rich theme + components
│   │
│   ├── commands/
│   │   ├── detect.py
│   │   ├── export.py
│   │   ├── merge.py
│   │   ├── info.py
│   │   ├── usage.py
│   │   ├── about.py
│   │   ├── credits.py
│   │   └── changelog.py
│   │
│   └── core/
│       ├── binaries.py         ffmpeg / ffprobe resolution
│       ├── keyframes.py        keyframe extraction
│       ├── video.py            metadata + scene merging
│       ├── segmenter.py        ffmpeg segment runner
│       ├── thumbnails.py       thumbnail generation
│       ├── similarity.py       cosine similarity check
│       ├── hevc.py             HEVC codec detection
│       ├── image.py            image crop with GIF support
│       └── detection/
│           ├── keyframe.py
│           └── edge.py
│
├── pyproject.toml
└── README.md
```

---

## Install

```bash
pip install amverge

# Edge detection method (requires OpenCV):
pip install amverge[edge]
```

Requires **ffmpeg** and **ffprobe** on your PATH (or drop them in the working directory).

---

## Getting Started

### Interactive mode

```bash
amverge
```

Launches a full wizard session — pick a command, fill in options step by step.

### Direct commands

```bash
amverge detect episode.mp4
amverge detect episode.mp4 --method edge --min-duration 0.5
amverge export episode.mp4 --scenes episode_scenes/scenes.json
amverge export episode.mp4 --scenes scenes.json --select 0,2,5-8 --merge
amverge merge scene_0001.mp4 scene_0002.mp4 --output out.mp4
amverge info episode.mp4
```

### Info

```bash
amverge usage      # command reference + examples
amverge about      # what this is
amverge credits    # contributors
amverge changelog  # version history
```

---

## CLI Reference

### `amverge detect`

| Flag | Default | Description |
|------|---------|-------------|
| `--output / -o` | `<name>_scenes/` | Output directory |
| `--method / -m` | `keyframe` | `keyframe` or `edge` |
| `--format / -f` | `table` | `table`, `json`, or `paths` |
| `--json-output` | — | Also write JSON to a file |
| `--no-thumbnails` | false | Skip thumbnail generation |
| `--no-similarity` | false | Skip similarity check |
| `--min-duration` | `0.25` | Merge scenes shorter than N seconds |
| `--workers` | `4` | Thumbnail thread count |
| `--similarity-threshold` | `0.10` | Similarity cutoff (lower = stricter) |
| `--edge-threshold` | `0.15` | Edge detection sensitivity |
| `--edge-radius` | `0.6` | Keyframe window radius |

### `amverge export`

| Flag | Default | Description |
|------|---------|-------------|
| `--scenes / -s` | required | `scenes.json` from `detect` |
| `--output / -o` | `./export` | Output directory |
| `--select` | all | Index range: `0,2,5-8` |
| `--merge` | false | Merge selection into one file |
| `--codec` | `copy` | `copy`, `h264`, `hevc` |

### `amverge merge`

```bash
amverge merge clip1.mp4 clip2.mp4 clip3.mp4 --output merged.mp4
```

### `amverge info`

```bash
amverge info episode.mp4
```

---

## Detection Methods

| Method | Speed | Accuracy | Requirement |
|--------|-------|----------|-------------|
| `keyframe` | Fast | Cuts at I-frame boundaries | base install |
| `edge` | Slower | Frame-accurate via Canny edges | `pip install amverge[edge]` |

Use `keyframe` for most anime and broadcast content.  
Use `edge` for heavily compressed files where keyframe placement is unreliable.

---

## Python Library

```python
from amverge_cli import detect_scenes

result = detect_scenes("episode.mp4")

for scene in result.scenes:
    print(scene.index, scene.start, scene.end, scene.path)

for a, b in result.similar_pairs:
    print(f"Scenes {a} and {b} look similar")
```

### `detect_scenes()` signature

```python
result = detect_scenes(
    video_path="episode.mp4",
    output_dir="./scenes",        # defaults to <name>_scenes/ next to video
    method="keyframe",             # "keyframe" or "edge"
    min_duration=0.25,             # merge scenes shorter than N seconds
    thumbnails=True,
    similarity=True,
    similarity_threshold=0.10,
    thumbnail_workers=4,
    edge_threshold=0.15,
    edge_radius=0.6,
    progress=lambda stage, pct, msg: print(f"[{stage}] {pct}%"),
)
```

### Low-level modules

```python
from amverge_cli.core.binaries   import get_ffmpeg, get_ffprobe
from amverge_cli.core.keyframes  import generate_keyframes
from amverge_cli.core.video      import get_video_info
from amverge_cli.core.segmenter  import run_ffmpeg_segment
from amverge_cli.core.thumbnails import generate_thumbnails
from amverge_cli.core.similarity import find_similar_pairs
from amverge_cli.core.hevc       import is_hevc
from amverge_cli.core.image      import crop_image, CropData
from amverge_cli.core.detection  import detect_cuts_by_keyframe, detect_cuts_by_edge
```

---

## Credits

Built by [Moongetsu](https://github.com/Moongetsu) as a standalone port of the [AMVerge](https://github.com/crptk/AMVerge) backend.

Original AMVerge by [Crptk](https://github.com/crptk) and contributors.

---

## License

MIT
