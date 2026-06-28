# Contributing

Contributions are welcome. Bug fixes, new features, and improvements to detection accuracy are all fair game.

---

## Setup

```bash
git clone https://github.com/moongetsu/AMVerge-CLI
cd AMVerge-CLI
pip install -e ".[edge]"
```

---

## Project Structure

```txt
amverge/
├── cli.py          entry point + command registration
├── pipeline.py     high-level detect_scenes() API
├── wizard.py       interactive session (no-args mode)
├── ui.py           shared Rich theme + console helpers
│
├── commands/       one file per CLI subcommand
│   ├── detect.py
│   ├── export.py
│   ├── merge.py
│   ├── info.py
│   ├── usage.py
│   ├── about.py
│   ├── credits.py
│   └── changelog.py
│
└── core/           pure logic, no CLI dependencies
    ├── binaries.py
    ├── keyframes.py
    ├── video.py
    ├── segmenter.py
    ├── thumbnails.py
    ├── similarity.py
    ├── hevc.py
    ├── image.py
    └── detection/
        ├── keyframe.py
        └── edge.py
```

---

## Guidelines

- Keep `core/` modules free of CLI/Rich dependencies - they are importable as a library
- New CLI commands go in `commands/` and get registered in `cli.py` and added to the wizard in `wizard.py`
- Match the existing commit style: `(add)`, `(fix)`, `(update)` prefix

---

## Links

- Main repo: [github.com/crptk/AMVerge](https://github.com/crptk/AMVerge)
- Discord: [discord.gg/bmXjTgsAaN](https://discord.gg/bmXjTgsAaN)
