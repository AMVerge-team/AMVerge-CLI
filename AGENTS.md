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
pip install -e ".[ml]"     # + TransNetV2 ML detection (torch, GPU optional)
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
| Video decode | PyAV (packet demux + keyframe timestamps) |
| Video decode (ML) | Nelux (Windows native, optional) or FFmpeg pipe |
| Video process | FFmpeg / FFprobe (subprocess) |
| Scene detection | TransNetV2 via `transnetv2_pytorch` (optional, `[ml]` extra) |
| Scene cutting | Smart cut: lossless copy + re-encode tail, or full re-encode (HEVC/fallback) |
| Image | Pillow |
| Numerics | NumPy |
| GPU | PyTorch (CUDA auto-detected, CPU fallback) |
| Edge detection | OpenCV (optional, `[edge]` extra) |
| Package | hatchling, PyPI name `amverge` |
| Discord RPC | pypresence (optional, `[discord]` extra) |

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
│   │   ├── about/
│   │   │   ├── about.py         amverge about
│   │   │   ├── changelog.py     amverge changelog
│   │   │   ├── credits.py       amverge credits
│   │   │   └── usage.py         amverge usage  (CLI reference page)
│   │   ├── detection/
│   │   │   ├── bench.py         amverge bench  (keyframe scan + TransNetV2 decode/inference timing)
│   │   │   ├── cache.py         amverge cache  (list/clear TransNetV2 .npy scene caches)
│   │   │   ├── detect.py        amverge detect
│   │   │   ├── keyframes.py     amverge keyframes  (dump keyframe timestamps, --json, --count)
│   │   │   └── scenes.py        amverge scenes  (show scene list from .npy cache, --json, --min-duration)
│   │   ├── export/
│   │   │   ├── export.py        amverge export  (CODEC_PROFILES/AUDIO_FFMPEG dicts - wizard imports these)
│   │   │   └── merge.py         amverge merge
│   │   ├── upscaling/
│   │   │   ├── upscale.py       amverge upscale  (ml / anime4k / artcnn methods, --credits)
│   │   │   └── models.py        amverge models  (list/delete/download upscale model weights)
│   │   ├── info/
│   │   │   ├── info.py          amverge info  (stream metadata via PyAV)
│   │   │   └── probe.py         amverge probe  (V2 diagnostics: codec/HEVC/keyframes/scene cache)
│   │   ├── sidecar/
│   │   │   ├── backend.py       amverge backend <video> <output_dir>  (hidden - Rust sidecar replacement)
│   │   │   └── rpc_server.py    amverge rpc-server  (hidden - Discord RPC sidecar, reads JSON from stdin)
│   │   └── system/
│   │       ├── doctor.py        amverge doctor  (full health check: ffmpeg, deps, write access, pass/fail)
│   │       ├── gpu.py           amverge gpu  (PyTorch version, CUDA, GPU name/VRAM, all optional deps)
│   │       └── version.py       amverge version  (CLI + Python + all dep versions, --json for bug reports)
│   │
│   └── core/                pure logic - no Rich/Typer deps, safe as library
│       ├── codec/
│       │   └── codec_utils.py   check_if_hevc(), is_hevc(), CODEC_PROFILES, AUDIO_FFMPEG, CODEC_ALIASES, PRORES_CODECS, resolve_gpu()
│       ├── cutting/
│       │   ├── segmenter.py     run_ffmpeg_segment() - 1500-cut Windows chunking (V1)
│       │   └── smart_cut.py     cut_scene(), cut_all_scenes() - lossless copy / smartcut / reencode
│       ├── detection/
│       │   ├── edge.py          detect_cuts_by_edge() - guarded cv2 import (V1)
│       │   ├── keyframe.py      detect_cuts_by_keyframe() (V1)
│       │   ├── nelux_runtime.py _get_nelux_video_reader() - Windows DLL config for Nelux
│       │   └── scene_detection.py   decode_video_frames_nelux(), decode_and_detect_scenes(), run_model_one_pass()
│       ├── discord/
│       │   └── discord_rpc.py   DiscordRPC class - pypresence wrapper, CLIENT_ID from AMVerge
│       ├── image/
│       │   └── image.py         crop_image() + CropData - supports animated GIF
│       ├── infra/
│       │   ├── binaries.py      get_binary(), get_ffmpeg(), get_ffprobe() - PyInstaller-aware PATH search
│       │   ├── diagnostics.py   get_gpu_info(), get_versions() - clean wrappers
│       │   └── ipc.py           emit_progress(), emit_event(), log(), check_if_path_exists(), build_video_cache_prefix()
│       ├── keyframes/
│       │   ├── keyframe_align.py    get_keyframe_timestamps_pyav(), classify_scenes_by_keyframe_alignment()
│       │   └── keyframes.py         generate_keyframes() - PyAV packet demux (V1 detect command)
│       ├── similarity/
│       │   └── similarity.py    find_similar_pairs() - cosine similarity on pixel arrays
│       ├── thumbnails/
│       │   ├── thumbnails.py            make_thumbnail(), generate_thumbnails() - ThreadPoolExecutor
│       │   └── thumbnails_streaming.py  streaming thumbnail gen with IPC events (V1 backend mode)
│       ├── transnet/
│       │   └── transnet_constants.py    FRAME_WIDTH/HEIGHT/CHANNELS/BYTES, WINDOW_SIZE, STRIDE
│       ├── upscaling/
│       │   ├── registry.json       declarative model registry - add models here, CLI auto-discovers
│       │   ├── registry.py          loads registry.json, builds URLs, query functions
│       │   ├── engine.py            upscale_model() - dispatches ml (spandrel) / shader / onnx; owns ml path only
│       │   ├── anime4k.py           upscale_video_anime4k() - real Anime4K GLSL via libplacebo, lanczos fallback
│       │   ├── artcnn.py            upscale_video_artcnn() - ArtCNN ONNX inference (luma 2x), download helpers
│       │   ├── ffmpeg_helpers.py    shared: mux_audio(), build_ffmpeg_pipe(), get_video_dims_ffprobe(), CREATE_NO_WINDOW
│       │   ├── monitor.py           SystemMonitor - GPU/CPU/RAM sampling + ETA during upscale
│       │   ├── __init__.py          exports: UPSCALE_REGISTRY, upscale_model, download_*, is_*_downloaded, ...
│       │   └── weight_loader.py     download_weights(), verify_weight_hash(), load_weights_if_available() (ml .pth only)
│       ├── video/
│       │   ├── probe_utils.py   probe_video_fps/duration/dimensions/total_frames via ffprobe
│       │   ├── scene_utils.py   scenes_to_objects(), scenes_frames_to_seconds()
│       │   └── video.py         get_video_duration(), get_video_info(), merge_short_scenes()
│       └── wrappers/
│           ├── amverge_video.py     AmvergeVideo - unified class wrapping video metadata, cutting, thumbnails, detection
│           ├── image_crop.py        ImageCrop - renamed from CropData with apply() method
│           ├── scene_cache.py       SceneCache - unified .npy cache save/load/list/clear
│           ├── scene_detector.py    SceneDetector - unified class wrapping all detection methods
│           ├── scene_exporter.py    SceneExporter - unified class wrapping export/encode/merge
│           ├── similarity_checker.py    SimilarityChecker - unified class wrapping pair similarity detection
│           ├── thumbnail_generator.py   ThumbnailGenerator - unified class wrapping thumbnail generation
│           └── transnet_config.py   TransNetConfig - frozen dataclass wrapping TransNetV2 constants
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

### V2 backend pipeline (TransNetV2)

```
decode_video_frames_nelux() or decode_and_detect_scenes()
        ↓ (frames ndarray)
run_model_one_pass() (TransNetV2, GPU/CPU)
        ↓ (scenes_secs, scenes_frames ndarray - cached as .npy)
scenes_to_objects()
        ↓ emit INITIAL_CLIPS_READY|[json]
get_keyframe_timestamps_pyav() + check_if_hevc()
        ↓
classify_scenes_by_keyframe_alignment()
        ↓
Phase 1: cut_all_scenes() lossless copy (max_workers=8)  -> emit CLIP_READY per scene
        ↓ emit PHASE1_COMPLETE
Phase 2: cut_all_scenes() re-encode (max_workers=2)      -> emit REENCODE_PROGRESS + CLIP_READY
```

### V1 detection pipeline (keyframe, no ML)

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
| `core/cutting/segmenter.py` | Windows 32,767-char command line limit. If video has >1500 cut points, chunks into multiple ffmpeg passes. Do not remove this chunking. |
| `core/keyframes/keyframes.py` | Fast path reads packet metadata only (no frame decode). Falls back to full decode for pathological encodes. Deduplicates I-frames within short windows. |
| `core/detection/edge.py` | `import cv2` is inside the function body, not at module level. Raises clear `ImportError` pointing to `pip install amverge[edge]` if OpenCV missing. Keep it this way - edge is an optional dep. |
| `wizard.py` | `_credits_table()` is imported from `commands/about/credits.py` to avoid duplication. `_wizard_export()` imports `CODEC_PROFILES`, `AUDIO_FFMPEG`, `CODEC_ALIASES`, `PRORES_CODECS`, `resolve_gpu` from `core/codec/codec_utils.py` - single source of truth for codec mappings. |
| `ui.py` | `err` console (stderr) used for all interactive/wizard output. `console` (stdout) for command results. Do not mix them. `ok()`/`warn()`/`fail()` use ASCII-safe marker `>` - Python `●`/`→` crash on CP1252 Windows terminals. |
| `core/similarity/similarity.py` | `find_similar_pairs()` accepts both `scene_index` and `index` keys for V1 (collect_scenes) / V2 (Scene.to_dict()) compat. |
| `commands/export/export.py` | `CODEC_PROFILES`/`AUDIO_FFMPEG`/`CODEC_ALIASES`/`PRORES_CODECS`/`_resolve_gpu` imported from `core/codec/codec_utils.py`. |
| `pipeline.py` | `DetectionMethod` is `Literal["keyframe", "edge", "transnetv2"]`. TransNetV2 path uses `decode_and_detect_scenes()` + `cut_all_scenes()` (V2 pipeline). Monkey-patches `emit_progress` on `scene_detection`/`smart_cut` module-local refs (not `ipc` module) to route IPC progress to Rich callback. |
| `core/infra/ipc.py` | IPC protocol for Tauri app. V2 events: `PROGRESS\|pct\|msg`, `INITIAL_CLIPS_READY\|json`, `CLIP_READY\|idx\|path\|mode`, `PHASE1_COMPLETE`, `REENCODE_PROGRESS\|done\|total`. stdout reserved for final JSON. Never mix IPC output with Rich output. |
| `core/detection/scene_detection.py` | TransNetV2 inference. Requires `[ml]` extra. `TRANSNET_AVAILABLE` flag guards import at module level - raises clear `ImportError` if missing. Do not import torch at module level in other files. |
| `core/cutting/smart_cut.py` | Four cut modes: `copy` (start on keyframe), `snapped_copy` (HEVC CPU - snaps to nearest keyframe within 5s), `smartcut` (H.264 - encode tiny head + lossless tail), `reencode` (full fallback). Never remove the HEVC CPU path - HEVC re-encode without CUDA takes 10+ minutes. |
| `core/codec/codec_utils.py` | `check_if_hevc()` via ffprobe. Also contains `CODEC_PROFILES` (14 codec -> ffmpeg encoder mappings), `AUDIO_FFMPEG` (10 audio choices), `CODEC_ALIASES`, `PRORES_CODECS`, `resolve_gpu()`, `is_hevc()`. Single source of truth - `commands/export/export.py` and `wizard.py` import from here. Do not duplicate these dicts. |
| `core/detection/nelux_runtime.py` | Windows DLL setup for Nelux video reader. Set `AMVERGE_FFMPEG_BIN` env var to FFmpeg shared DLL directory. Idempotent - safe to call multiple times. |
| `core/keyframes/keyframe_align.py` | `get_keyframe_timestamps_pyav` uses PyAV demux with `type(stream.discard).nonkey` enum (PyAV 17.x; was `"NONKEY"` string in older PyAV). `classify_scenes_by_keyframe_alignment` partitions scenes for Phase 1 vs Phase 2 cutting. |
| `core/thumbnails/thumbnails_streaming.py` | V1 backend mode only. Emits events as each thumbnail completes. Not used in V2 backend. |
| `core/discord/discord_rpc.py` | Uses same CLIENT_ID as AMVerge app (`1497922104065134823`). Silently no-ops if pypresence not installed. `--no-rpc` flag on detect/export/merge to disable. Methods: idle/detecting/selecting/navigating/exporting/merging/complete/error. |
| `core/upscaling/engine.py` | `upscale_model(key, ...)` - unified dispatch. Reads model from registry, routes ml → `_upscale_ml()` (spandrel, in-file), shader → `anime4k.upscale_video_anime4k()`, onnx → `artcnn.upscale_video_artcnn()`. Engine owns the ml path only; shader/onnx live in their own modules. |
| `core/upscaling/registry.json` | Declarative model registry. To add a model, add one JSON entry with method, name, scales, credit, description, file/hash. CLI auto-discovers everything. `_source` holds the ml/anime4k/artcnn base URLs; `registry.py` builds per-model `url` (ml/onnx) or `download_url` (shader zip). |
| `core/upscaling/ffmpeg_helpers.py` | Shared FFmpeg utilities used by engine/anime4k/artcnn (avoids circular import): `mux_audio()` (copies source audio with `-c:a copy`, AAC re-encode only as fallback), `build_ffmpeg_pipe()` (rawvideo stdin pipe), `get_color_args()` (probe + pass through source color primaries/transfer/matrix/range), `get_video_dims_ffprobe()`, `encode_thread_count()`, `ensure_ffmpeg()`, `CREATE_NO_WINDOW`. All encoders use `-profile:v high` with NO explicit `-level` (x264 auto-selects; hardcoding 5.1 produced non-compliant streams for 4x-of-HD outputs). |
| `core/upscaling/weight_loader.py` | Downloads ml `.pth` weights to `models/upscale/<key>/<file>`. Resume support (HTTP Range), SHA-256 integrity verification, 3 retries. ml-only; ONNX downloads live in `artcnn.py`. |
| `core/upscaling/anime4k.py` | `upscale_video_anime4k()` - REAL Anime4K. Downloads v4.0.1 GLSL shaders (`models/upscale/anime4k/`), concatenates the mode's shader chain into `_chain_<mode>_<scale>x.glsl`, applies via FFmpeg `libplacebo=custom_shader_path`. Auto-detects libplacebo (`ffmpeg -filters`); falls back to lanczos+unsharp if absent. Modes light/medium/strong = Upscale_CNN_x2_{S,M,VL} (+ Restore_CNN for medium/strong); scale=4 adds a 2nd upscale pass. **Two FFmpeg gotchas (do not reintroduce):** (1) NEVER pass global `-init_hw_device vulkan` - it breaks libplacebo output negotiation (EINVAL); libplacebo self-inits its Vulkan device. (2) `custom_shader_path` cannot take an absolute Windows path (drive colon = filtergraph option separator, no escaping survives). The chain is staged into the OUTPUT file's dir, ffmpeg runs with `cwd=that dir`, path passed as bare basename, file deleted after. |
| `core/upscaling/artcnn.py` | `upscale_video_artcnn()` - ArtCNN ONNX (luma-only 2x doublers). Downloads to `models/upscale/artcnn/<file>` (single dir - was a path-mismatch bug vs weight_loader's per-key dir). Per-frame: BGR→YUV, run Y `[1,1,H,W]` → `[1,1,2H,2W]`, lanczos-upscale U/V, recombine, rawvideo pipe. Session uses `enable_cpu_mem_arena=False` + per-frame `del`/`gc` (HD full-frame inference OOMs otherwise). **Chroma models** (entry has `chroma_file`, e.g. `R8F64_Chroma`): loads a 2nd session; chroma net takes `[1,3,H,W]` (Y_2x + bilinear-upscaled U/V) → `[1,2,H,W]` reconstructed U/V, replacing the lanczos chroma path. `download_artcnn` fetches all `_model_files(entry)`; `is_artcnn_downloaded` checks all. Input/output tensor names auto-detected. v1.6.2 assets verified. Credit: ArtCNN by Artoriuz. |
| `commands/sidecar/rpc_server.py` | Hidden sidecar: `amverge rpc-server`. Long-lived process; Rust spawns it once and sends JSON commands via stdin (`{"type":"update","details":"...","state":"..."}`, `{"type":"clear"}`, `{"type":"shutdown"}`). Throttles Discord updates to max 1 per 15s. Exits when stdin closes or parent dies. |
| `commands/sidecar/backend.py` | V2 backend. Positional interface: `amverge backend <video_path> <output_dir> [import_method]`. Rust replaces `python app.py <video> <dir>` with `amverge backend <video> <dir>` - no Rust changes needed. Emits V2 IPC events. Outputs JSON schema v1.0 with `schema_version`, `run_id`, `video` metadata block. |

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
