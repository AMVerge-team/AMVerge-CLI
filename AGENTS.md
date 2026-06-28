# AGENTS.md - AMVerge CLI

AMVerge features as a CLI tool and Python library. Ports the AMVerge desktop app backend (by Crptk) into a standalone `pip install amverge` package.

## AI Agent Instructions

- **Update this file** when adding/removing files, changing architecture, adding new commands, or introducing conventions.
- **Commit style:** prefix tag in parentheses: `(add)` new features, `(fix)` bug fixes, `(update)` refactors/formatting/chores. Example: `(fix) wizard: handle KeyboardInterrupt during path input`.
- **Commit author:** always commit under the user's account. Do NOT add a `Co-Authored-By: Claude` trailer or any self-attribution.
- **Commit per task:** separate commit after each task/logical change. Do not batch unrelated changes.
- **No code comments:** do NOT add comments when writing or modifying code. Put any needed explanation in the commit message or this file.
- **No em dashes:** never use `—` in any prose, docs, README, or commit messages. Use a comma, colon, parentheses, or plain hyphen `-` instead.
- **Rich style= params:** theme names (`accent`, `muted`, etc.) only resolve inside markup strings `[accent]text[/]`. They do NOT work in `style=`, `header_style=`, or `title_style=` kwargs - use literal hex `#22c55e bold` there instead.

## Build & Run

```bash
pip install -e .           # base install (keyframe detection only)
pip install -e ".[edge]"   # + OpenCV for edge detection method
```

No build step. Pure Python package, `hatchling` backend.

```bash
amverge                    # interactive wizard (no-args mode)
amverge detect video.mp4   # direct command
amverge --help             # Typer help
```

To publish to PyPI (not done yet - verify `pypi.org/project/amverge` name is free first):

```bash
pip install build twine
python -m build
twine upload dist/*
```

## Tech Stack

| Layer | Tech |
|---|---|
| CLI | Typer |
| UI | Rich (custom green theme) |
| Video decode | PyAV (packet demux + frame decode) |
| Video process | FFmpeg / FFprobe (subprocess) |
| Image | Pillow |
| Numerics | NumPy |
| Edge detection | OpenCV (optional, `[edge]` extra) |
| Package | hatchling, PyPI name `amverge` |

## Directory Map

```
AMVerge-CLI/
├── amverge/
│   ├── __init__.py          public exports: detect_scenes, DetectResult, Scene, DetectionMethod
│   ├── __version__.py       version string
│   ├── cli.py               Typer app, registers all commands, no-args -> wizard
│   ├── pipeline.py          high-level detect_scenes() public API
│   ├── wizard.py            interactive session (no-args mode)
│   ├── ui.py                shared Rich theme, console, banner, progress, table helpers
│   │
│   ├── commands/
│   │   ├── detect.py        amverge detect
│   │   ├── export.py        amverge export
│   │   ├── merge.py         amverge merge
│   │   ├── info.py          amverge info
│   │   ├── usage.py         amverge usage  (CLI reference page)
│   │   ├── about.py         amverge about
│   │   ├── credits.py       amverge credits  (exports _credits_table() for wizard reuse)
│   │   └── changelog.py     amverge changelog
│   │
│   └── core/                pure logic - no Rich/Typer deps, safe as library
│       ├── binaries.py      get_ffmpeg(), get_ffprobe() - PATH then local bin/ fallback
│       ├── keyframes.py     generate_keyframes() - PyAV packet demux + decode fallback
│       ├── video.py         get_video_duration(), get_video_info(), merge_short_scenes()
│       ├── segmenter.py     run_ffmpeg_segment() - 1500-cut Windows chunking
│       ├── thumbnails.py    make_thumbnail(), generate_thumbnails() - ThreadPoolExecutor
│       ├── similarity.py    find_similar_pairs() - cosine similarity on pixel arrays
│       ├── hevc.py          is_hevc() - ffprobe codec check
│       ├── image.py         crop_image() + CropData - supports animated GIF
│       └── detection/
│           ├── keyframe.py  detect_cuts_by_keyframe()
│           └── edge.py      detect_cuts_by_edge() - guarded cv2 import, clear error if missing
│
├── docs/
│   ├── installation.md
│   ├── cli-reference.md
│   ├── library.md
│   ├── detection-methods.md
│   └── contributing.md
│
├── assets/
│   └── amverge_title_gif.gif
│
├── pyproject.toml
├── README.md
└── AGENTS.md
```

## Key Architecture

### No-args wizard routing

`cli.py` uses `@app.callback(invoke_without_command=True)`. When `ctx.invoked_subcommand is None`, calls `run_wizard()` from `wizard.py`. All wizard output goes to `stderr` so stdout stays clean for piping.

### Detection pipeline

```
generate_keyframes() (PyAV packet demux)
        ↓
merge_short_scenes() (drop cuts < min_duration)
        ↓
run_ffmpeg_segment() (ffmpeg -segment_times, stream copy)
        ↓
generate_thumbnails() (PyAV decode, ThreadPoolExecutor)
        ↓
find_similar_pairs() (cosine similarity on pixel arrays)
```

### Library API

```python
from amverge import detect_scenes

result = detect_scenes("episode.mp4")
for scene in result.scenes:
    print(scene.index, scene.start, scene.end, scene.path)
```

`core/` has no CLI dependencies. Import anything from it without pulling in Rich or Typer.

## Code Conventions

- **Python:** snake_case vars/fns, PascalCase dataclasses, type hints on all public APIs
- **CLI commands:** one file per command in `commands/`, registered in `cli.py`, added to wizard in `wizard.py`
- **Library modules:** all in `core/`, no Rich/Typer imports allowed
- **UI:** all Rich output through `ui.py` helpers (`console`, `err`, `banner()`, `make_table()`)
- **Subprocess:** all ffmpeg/ffprobe calls include `creationflags=0x08000000` on win32 (suppress console popups)

## Critical Paths

| File | Role |
|---|---|
| `core/segmenter.py` | Windows 32,767-char command line limit. If video has >1500 cut points, chunks into multiple ffmpeg passes. Do not remove this chunking. |
| `core/keyframes.py` | Fast path reads packet metadata only (no frame decode). Falls back to full decode for pathological encodes. Deduplicates I-frames within short windows. |
| `core/detection/edge.py` | `import cv2` is inside the function body, not at module level. Raises clear `ImportError` pointing to `pip install amverge[edge]` if OpenCV missing. Keep it this way - edge is an optional dep. |
| `wizard.py` | `_credits_table()` is imported from `commands/credits.py` to avoid duplication. The wizard and the direct `amverge credits` command share one table definition. |
| `ui.py` | `err` console (stderr) used for all interactive/wizard output. `console` (stdout) for command results. Do not mix them. |

## Theme

```python
THEME = Theme({
    "accent":        "#22c55e",
    "accent.bright": "#00f07a",
    "muted":         "bright_black",
    "success":       "#22c55e bold",
    "warn":          "#facc15",
    "error":         "#ef4444",
    "label":         "white",
    "bar.back":      "bright_black",
    "bar.complete":  "#22c55e",
    "bar.finished":  "#00f07a",
}, inherit=False)
```

Banner markup: `[accent]AMV[/][white bold]erge[/]` - AMV is green, erge is white. Match this everywhere.

## Origin

All core logic ported from `AMVergeNew/backend/` (Crptk's original AMVerge desktop app).
Color palette from `AMVergeNew/frontend/src/styles/variables.css`.
