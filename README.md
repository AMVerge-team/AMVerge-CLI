<p align="center">
  <img src="assets/amverge_title_gif.gif" alt="AMVerge CLI" width="1440"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue?style=flat-square" alt="Python"/>
  <img src="https://img.shields.io/badge/pypi-amverge-22c55e?style=flat-square" alt="PyPI"/>
  <img src="https://img.shields.io/badge/license-GPL--3.0-22c55e?style=flat-square" alt="License"/>
</p>

# AMVerge CLI

**AMVerge Features as a CLI Tool and Python Library.**  
Port of the AMVerge desktop app backend by [Crptk](https://github.com/crptk). Split videos into scenes, export clips, merge fragments, and build your own tools on top of it.

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

Detection extracts keyframe timestamps via PyAV, filters short scenes, then segments using `ffmpeg -segment_times` with stream copy.  
Thumbnails are decoded via PyAV and written as JPEG in parallel.  
Similarity compares adjacent thumbnails using cosine similarity on pixel arrays.

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
│   ├── commands/               one file per CLI subcommand
│   └── core/                   pure logic, no CLI dependencies
│
├── docs/
│   ├── installation.md
│   ├── cli-reference.md
│   ├── library.md
│   ├── detection-methods.md
│   └── contributing.md
│
├── assets/
├── pyproject.toml
└── README.md
```

---

## Install

```bash
pip install amverge
```

See [docs/installation.md](docs/installation.md) for FFmpeg setup, optional dependencies, and dev install.

---

## Quick Start

```bash
# Interactive wizard
amverge

# Direct commands
amverge detect episode.mp4
amverge export episode.mp4 --scenes episode_scenes/scenes.json --select 0,2,5-8
amverge merge clip1.mp4 clip2.mp4 --output out.mp4
amverge info episode.mp4
```

```python
from amverge_cli import detect_scenes

result = detect_scenes("episode.mp4")
for scene in result.scenes:
    print(scene.index, scene.start, scene.end, scene.path)
```

---

## Documentation

| | |
|---|---|
| [Installation](docs/installation.md) | Requirements, FFmpeg setup, optional deps, dev install |
| [CLI Reference](docs/cli-reference.md) | All commands, flags, and usage examples |
| [Python Library](docs/library.md) | API reference, return types, low-level modules |
| [Detection Methods](docs/detection-methods.md) | Keyframe vs edge, tuning parameters |
| [Contributing](docs/contributing.md) | Project structure, guidelines, links |

---

## AI Agents

An [AGENTS.md](AGENTS.md) file is included for AI coding assistants (OpenCode, Claude Code, Cursor, etc.).

That said - using AI to work on this without understanding the codebase is not recommended. You can save time with AI tools, but you need to know what you are doing first. Read the code, understand the architecture, then use the agents file if it helps your workflow.

---

## Credits

Built by [Moongetsu](https://github.com/Moongetsu) as a standalone port of the [AMVerge](https://github.com/crptk/AMVerge) backend.

AMVerge was created by [Crptk](https://github.com/crptk). All core scene detection and clip management logic originates from the original AMVerge project.

---

## License

AMVerge CLI is licensed under the GNU GPL v3.0.

Any derivative work must also be open-source under the same license.
