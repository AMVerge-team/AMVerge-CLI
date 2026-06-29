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
├── __init__.py              public exports: detect_scenes, DetectResult, Scene, DetectionMethod
├── __version__.py           version string
├── cli.py                   Typer app, registers commands, no-args -> wizard
├── pipeline.py              high-level detect_scenes() API
├── wizard.py                interactive session (no-args mode)
├── ui.py                    shared Rich theme, console, banner, progress, table helpers
│
├── commands/                one file per CLI subcommand
│   ├── about.py             amverge about
│   ├── backend.py           amverge backend (hidden - Rust sidecar replacement)
│   ├── bench.py             amverge bench (keyframe scan + TransNetV2 timing)
│   ├── cache.py             amverge cache (list/clear .npy scene caches)
│   ├── changelog.py         amverge changelog
│   ├── credits.py           amverge credits
│   ├── detect.py            amverge detect
│   ├── doctor.py            amverge doctor (full health check)
│   ├── export.py            amverge export (codec profiles + hardware selection)
│   ├── gpu.py               amverge gpu (CUDA, GPU name/VRAM, optional deps)
│   ├── info.py              amverge info (stream metadata via PyAV)
│   ├── keyframes.py         amverge keyframes (dump keyframe timestamps)
│   ├── merge.py             amverge merge
│   ├── probe.py             amverge probe (codec/HEVC/keyframes/scene cache diagnostics)
│   ├── rpc_server.py        amverge rpc-server (hidden - Discord RPC sidecar)
│   ├── scenes.py            amverge scenes (show scene list from .npy cache)
│   ├── usage.py             amverge usage (CLI reference page)
│   └── version.py           amverge version (CLI + Python + dep versions)
│
├── core/                    pure logic, no CLI/Rich deps
│   ├── binaries.py          get_binary(), get_ffmpeg(), get_ffprobe()
│   ├── codec_utils.py       check_if_hevc(), CODEC_PROFILES, AUDIO_FFMPEG, codec aliases
│   ├── diagnostics.py       get_gpu_info(), get_versions()
│   ├── discord_rpc.py       DiscordRPC class (pypresence wrapper)
│   ├── hevc.py              is_hevc() (V1 codec check)
│   ├── image.py             crop_image() + CropData
│   ├── ipc.py               emit_progress(), emit_event(), log(), IPC protocol
│   ├── keyframe_align.py    get_keyframe_timestamps_pyav(), classify_scenes_by_keyframe_alignment()
│   ├── keyframes.py         generate_keyframes() (V1 packet demux)
│   ├── nelux_runtime.py     _get_nelux_video_reader() (Windows DLL config)
│   ├── probe_utils.py       probe_video_fps/duration/dimensions/total_frames via ffprobe
│   ├── scene_detection.py   decode_video_frames_nelux(), decode_and_detect_scenes(), run_model_one_pass()
│   ├── scene_utils.py       scenes_to_objects(), scenes_frames_to_seconds()
│   ├── segmenter.py         run_ffmpeg_segment() (1500-cut Windows chunking)
│   ├── similarity.py        find_similar_pairs() (cosine similarity)
│   ├── smart_cut.py         cut_scene(), cut_all_scenes() (lossless copy / smartcut / reencode)
│   ├── thumbnails.py        make_thumbnail(), generate_thumbnails() (ThreadPoolExecutor)
│   ├── thumbnails_streaming.py  streaming thumbnail gen with IPC events (V1)
│   ├── transnet_constants.py    FRAME_WIDTH/HEIGHT/CHANNELS, WINDOW_SIZE, STRIDE
│   ├── video.py             get_video_duration(), get_video_info(), merge_short_scenes()
│   ├── detection/
│   │   ├── keyframe.py      detect_cuts_by_keyframe() (V1)
│   │   └── edge.py          detect_cuts_by_edge() (guarded cv2 import, V1)
│   └── ...
│
├── examples/                runnable Python scripts
│   ├── custom-pipeline/     full end-to-end pipeline
│   ├── cutting/             smart cut, ffmpeg segment
│   ├── detect/              keyframe, edge, TransNetV2 detection
│   ├── diagnostics/         GPU, CUDA, dependency versions
│   ├── discord-rpc/         Discord Rich Presence
│   ├── export/              copy, re-encode with profiles, merge
│   ├── info-probe/          stream metadata, diagnostics, HEVC check
│   ├── keyframes/           extraction + classification for cutting
│   ├── similarity/          adjacent scene similarity detection
│   └── thumbnails/          JPEG thumbnail generation
│
├── docs/                    markdown documentation
├── assets/                  GIF and image assets
├── pyproject.toml
├── README.md
└── AGENTS.md
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

- Main repo: [github.com/AMVerge-team/AMVerge](https://github.com/AMVerge-team/AMVerge)
- Discord: [discord.gg/bmXjTgsAaN](https://discord.gg/bmXjTgsAaN)
