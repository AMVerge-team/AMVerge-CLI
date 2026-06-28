# Contributing

Contributions are welcome. Bug fixes, new features, and improvements to detection accuracy are all fair game.

---

## Setup

```bash
git clone https://github.com/AMVerge-team/AMVerge-CLI
cd AMVerge-CLI
pip install -e ".[ml,edge,discord]"
```

---

## Project Structure

```txt
amverge/
├── cli.py              entry point + command registration
├── pipeline.py         high-level detect_scenes() API
├── wizard.py           interactive session (no-args mode)
├── ui.py               shared Rich theme + console helpers
│
├── commands/           one file per CLI subcommand
│   ├── detect.py       export.py       merge.py       info.py
│   ├── probe.py        gpu.py          doctor.py      version.py
│   ├── bench.py        cache.py        keyframes.py   scenes.py
│   ├── backend.py      rpc_server.py   usage.py       about.py
│   ├── credits.py      changelog.py
│
├── core/               pure logic, no CLI/Rich deps
│   ├── binaries.py         scene_detection.py    smart_cut.py
│   ├── keyframe_align.py   keyframes.py          segmenter.py
│   ├── scene_utils.py      probe_utils.py        codec_utils.py
│   ├── video.py            thumbnails.py         similarity.py
│   ├── hevc.py             image.py              ipc.py
│   ├── discord_rpc.py      transnet_constants.py
│   └── detection/          keyframe.py  edge.py
│
├── examples/           20 runnable scripts in 10 categories
│   ├── detect/           export/         info-probe/
│   ├── keyframes/        cutting/        thumbnails/
│   ├── similarity/       diagnostics/    discord-rpc/
│   └── custom-pipeline/
│
docs/                   markdown documentation
assets/                 GIF and image assets
```

---

## Guidelines

- Keep `core/` modules free of CLI/Rich dependencies
- New CLI commands go in `commands/`, register in `cli.py`, add to wizard in `wizard.py`
- Match the existing commit style: `(add)`, `(fix)`, `(update)` prefix
- One commit per logical change
- No code comments unless asked
- No em dashes in prose or commit messages
- Update `AGENTS.md` when adding/removing files or changing architecture

---

## Links

- Main repo: [github.com/crptk/AMVerge](https://github.com/crptk/AMVerge)
- Discord: [discord.gg/bmXjTgsAaN](https://discord.gg/bmXjTgsAaN)
