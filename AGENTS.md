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

## Cross-Repo Contracts

Sibling repos sit next to this one: `../AMVerge` (Tauri desktop app) and `../AMVerge-Extension` (After Effects CEP panel). Each repo has its own agent, and this agent owns this repo only.

- **Reading** sibling repos needs no prompt. **Editing** them asks the user every time (`.claude/hooks/sibling_edit_guard.py`, PreToolUse, reads `siblings` from `contracts.json`). The hook covers the file-editing tools, not shell commands, so never change a sibling repo through Bash. Path-based `Edit(../...)` permission rules were tried and do not work: they cannot match outside the project root. Only do it for trivial fixes the user approves (typo, a one-line string rename). Anything else goes through `/handoff`.
- `.claude/contracts.json` lists the files here that other repos depend on, and the files there that consume them. Keep it current when a contract moves or a new consumer appears.
- `.claude/hooks/contract_guard.py` (PostToolUse) reminds you, once per session per contract, when you edit one of those files.
- If an edit changes anything consumers rely on (command or flag names, defaults, IPC event names or field order, JSON fields, extras), run `/handoff <repo>` before finishing. It writes a brief to `.claude/handoffs/` (gitignored), runs that repo's agent headless in its own checkout (accept-edits, cannot edit other repos, does not commit), and relays its report.
- This repo is upstream: AMVerge and AMVerge-Extension adapt to it. Prefer additive changes (a new flag or event) over renames so already-shipped app and extension builds keep working.

| Contract | Here | Consumed by |
|---|---|---|
| IPC events on stderr | `core/infra/ipc.py`, `core/infra/preview.py`, `core/thumbnails/thumbnails_streaming.py` | AMVerge `commands/scenes.rs`, `export.rs`, `scenepacks.rs`; Extension `js/utils/amvergeHandler.js`, `exportHandler.js` |
| Sidecar commands + backend JSON schema | `commands/sidecar/*`, `core/video/scene_utils.py` | AMVerge `commands/scenes.rs`, `scenepacks.rs` |
| `detect` flags + stdout JSON | `commands/detection/detect.py`, `pipeline.py` | Extension `js/utils/amvergeHandler.js` |
| `export`/`merge` flags, `--inputs-json`, codecs | `commands/export/*`, `core/export/*`, `core/codec/codec_utils.py` | AMVerge `commands/export*`, `src/features/export/*`; Extension `js/utils/exportHandler.js` |
| `--ipc` post passes, `models --json` | `commands/depth/*`, `deadframes/*`, `interpolation/*`, `upscaling/models.py` | AMVerge `commands/export.rs`, `models.rs` |
| Command names, extras, Python version | `cli.py`, `pyproject.toml` | AMVerge `commands/deps/*`; Extension `js/utils/amvergeRuntime.js`, `js/components/toolsSetup.js` |

## Build & Run

```bash
pip install -e .           # base install (keyframe detection only)
pip install -e ".[edge]"   # + OpenCV for edge detection method
pip install -e ".[ml]"     # + TransNetV2 ML detection (torch, GPU optional)
pip install -e ".[interpolation]"   # + RIFE PyTorch CUDA/CPU inference
pip install -e ".[flowframes]"      # no extra deps (external Flowframes.exe)
pip install -e ".[upscale]"         # + OpenCV, spandrel, onnxruntime for upscaling
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
| Scene cutting | Imports: keyframe detection stream-copies and TransNetV2 re-encodes exact boundaries. Exports choose `copy` (stream copy snapped outward to keyframes) or `reencode` (exact boundary). |
| Image | Pillow |
| Numerics | NumPy |
| GPU | PyTorch (CUDA auto-detected, CPU fallback) |
| Edge detection | OpenCV (optional, `[edge]` extra) |
| Package | hatchling, PyPI name `amverge` |
| Frame interpolation | Flowframes 1.42.0 (external .exe, Windows-only, NVIDIA GPU recommended; free 1.36.0 planned; `[flowframes]` extra) |
| Discord RPC | pypresence (optional, `[discord]` extra) |

## Directory Map

```
AMVerge-CLI/
├── amverge/
│   ├── __init__.py          public exports: detect_scenes, DetectResult, Scene, DetectionMethod, DecodeMethod
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
│   │   │   └── models.py        amverge models  (list/delete/download depth + interpolation model weights, --json for the app bridge)
│   │   ├── interpolation/
│   │   │   ├── interpolate.py       amverge interpolate  (Python RIFE inference)
│       │   │   ├── flowframes.py    amverge flowframes  (Flowframes 1.42.0 external process; free 1.36.0 planned)
│   │   │   └── flowframes_path.py   amverge flowframes-path  (set/show Flowframes.exe path)
│   │   ├── deadframes/
│   │   │   └── deadframes.py    amverge deadframes  (deadframe removal via optical flow + ORB homography)
│   │   ├── pipeline/
│   │   │   └── pipeline.py      amverge pipeline  (chain deadframes + upscale + interpolate, preset save/load)
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
│       │   └── smart_cut.py     cut_scene(), cut_all_scenes(), snap_range_to_keyframes() - copy (keyframe-snapped stream copy) or reencode (exact); export selects the mode
│       ├── detection/
│       │   ├── edge.py          detect_cuts_by_edge() - guarded cv2 import (V1)
│       │   ├── keyframe.py      detect_cuts_by_keyframe() (V1)
│       │   ├── nelux_runtime.py _get_nelux_video_reader() - Windows DLL config for Nelux
│       │   └── ai_scene_detection.py   decode_video_frames_nelux(), decode_and_detect_scenes(), run_model_one_pass()
│       ├── discord/
│       │   └── discord_rpc.py   DiscordRPC class - pypresence wrapper, CLIENT_ID from AMVerge
│       ├── image/
│       │   └── image.py         crop_image() + CropData - supports animated GIF
│       ├── infra/
│       │   ├── binaries.py      get_binary(), get_ffmpeg(), get_ffprobe() - PyInstaller-aware PATH search
│       │   ├── diagnostics.py   get_gpu_info(), get_versions() - clean wrappers
│       │   ├── ipc.py           emit_progress(), emit_event(), log(), check_if_path_exists(), build_video_cache_prefix()
│       │   └── preview.py        PreviewEmitter + ipc_callbacks() - throttled JPEG snapshots, emits PREVIEW_FRAME|tag|path|seq (ping-pongs 2 files/tag)
│       ├── keyframes/
│       │   ├── keyframe_align.py    get_keyframe_timestamps_pyav()
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
│       │   ├── monitor.py           SystemMonitor - GPU/CPU/RAM sampling + ETA during upscale and interpolation
│       │   ├── __init__.py          exports: UPSCALE_REGISTRY, upscale_model, download_*, is_*_downloaded, ...
│       │   └── weight_loader.py     download_weights(), verify_weight_hash(), load_weights_if_available() (ml .pth only)
│       ├── interpolation/
│       │   ├── rife_arch.py           RIFEModel + IFNet (light/heavy IFBlock channel widths)
│       │   ├── registry.json          declarative model registry - add models here, CLI auto-discovers
│       │   ├── registry.py             loads registry.json, builds URLs, query functions
│       │   ├── weight_loader.py        download_weights(), verify_weight_hash(), load_weights_if_available()
│       │   ├── engine.py               interpolate_video() - RIFE PyTorch CUDA/CPU inference
│       │   ├── flowframes.py           run_flowframes(), flowframes_available(), cancel_flowframes() - Flowframes 1.42.0 integration; free 1.36.0 planned
│       │   └── __init__.py             exports: interpolate_video, run_flowframes, INTERPOLATION_REGISTRY, download_weights, ...
│       ├── deadframes/
│       │   ├── registry.json          declarative model registry - add models here, CLI auto-discovers
│       │   ├── registry.py             loads registry.json, builds URLs, query functions
│       │   ├── weight_loader.py        download_weights(), verify_weight_hash()
│       │   ├── engine.py               DeadFrameDetector, run_deadframes() - optical flow + ORB homography + motion-area analysis
│       │   └── __init__.py             exports: run_deadframes, DEADFRAMES_REGISTRY, DEADFRAMES_AVAILABLE, ...
│       ├── pipeline/
│       │   └── presets.py       list/load/save/delete pipeline presets as JSON
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
├── .claude/
│   ├── settings.json        sibling read access, registers both hooks
│   ├── contracts.json       cross-repo contracts (see Cross-Repo Contracts)
│   ├── hooks/sibling_edit_guard.py   PreToolUse: ask before editing a sibling repo
│   ├── hooks/contract_guard.py   PostToolUse reminder to /handoff
│   └── skills/handoff/      /handoff skill: brief + headless sibling agent
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
cut_all_scenes() mode="reencode" (max_workers=2) -> emit CLIP_READY per scene
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

## Testing

```bash
pip install -e ".[dev,ml,edge]"
pytest                       # everything (~3 min, generates fixtures on first run)
pytest -m "not ml"           # skip TransNetV2 inference
pytest -m "not media"        # pure unit tests only, no ffmpeg needed
python -m tests.fixtures.generate   # pre-build the fixture videos
```

- **Fixture videos** are generated with ffmpeg, never committed. `tests/fixtures/specs.py` declares them (`MediaSpec`: codec, resolution, fps, per-scene frame counts, keyframe layout, audio tracks); `generate.py` renders each into `tests/fixtures/.media/` (override with `AMVERGE_TEST_MEDIA`), cached by a hash of the spec. Change a spec or bump `GENERATOR_VERSION` and it re-renders.
- **Formats:** the matrix covers H.264 and HEVC at 8 and 10 bit in both MP4 and MKV, AV1 at 8 bit (MP4) and 10 bit (MKV), and 10-bit ProRes in MOV; `test_fixture_matrix_covers_required_variety` enforces it. MKV fixtures always carry 1/1000 pts (`MediaSpec.timescale`), so their keyframe tolerances are half a millisecond. Export writes AVI, MP4, and MOV. Copy exports preflight every video stream and every copied audio stream through PyAV and FFmpeg target muxers before creating output. Copy output must keep the source's codec, profile and pix_fmt. A re-encoded preview keeps the source's bit depth (8 bit to yuv420p, 10 bit or more to yuv420p10le). Export re-encodes follow the profile's depth, except `av1_main`, which pins no pix_fmt and keeps the source's (`TestBitDepth`). The AV1 fixtures need `libsvtav1` in the ffmpeg build.
- **Content:** alternating solid blue / red scenes (blue first), a burnt-in timecode, and a 12-bit frame-number barcode strip along the bottom. `tests/fixtures/media.py` `read_frames()` decodes every frame back to `(number, color)`, so tests assert exact source frame numbers (catches dropped, duplicated, or bled frames), not just durations.
- **Matrix:** h264 / h265 (8-bit and main10) / ProRes 422 HQ and 4444; 23.976, 24, 25, 29.97, 30, 50, 60 fps; 270p to 1080p; keyframes forced at scene cuts, fixed GOP off the cuts (closed and open GOP), a single keyframe, or all-intra; no audio, mono, stereo, PCM, and two language-tagged tracks.
- **Known bugs** are `xfail(strict=True)` with the cause as the reason, all in `tests/fixtures/known_bugs.py`. When a fix lands the test XPASSes and fails the run, so delete the marker in the same commit.
- `conftest.py` sets `PYTORCH_ENABLE_MPS_FALLBACK=1` so TransNetV2 tests run on Apple Silicon regardless of import order.

## Code Conventions

- **Python:** snake_case vars/fns, PascalCase dataclasses, type hints on all public APIs
- **CLI commands:** one file per command in `commands/`, registered in `cli.py`, added to wizard in `wizard.py`
- **Library modules:** all in `core/`, no Rich/Typer imports allowed
- **UI:** all Rich output through `ui.py` helpers (`console`, `err`, `banner()`, `make_table()`)
- **Subprocess:** all ffmpeg/ffprobe calls include `creationflags=0x08000000` on win32 (suppress console popups)

## Critical Paths

| File | Role |
|---|---|
| `core/cutting/segmenter.py` | Windows 32,767-char command line limit. If video has >1500 cut points, chunks into multiple ffmpeg passes. Do not remove this chunking. A stream-copy split off a keyframe pushes forward to the next real one (same trade-off as `smart_cut`'s `copy` mode); the overshoot is left as visible content, not hidden or trimmed - `editlist.py` was retired for this reason (see `core/cutting/smart_cut.py` entry). |
| `core/keyframes/keyframes.py` | Fast path reads packet metadata only (no frame decode). Falls back to full decode for pathological encodes. Deduplicates I-frames within short windows. |
| `core/detection/edge.py` | `import cv2` is inside the function body, not at module level. Raises clear `ImportError` pointing to `pip install amverge[edge]` if OpenCV missing. Keep it this way - edge is an optional dep. |
| `wizard.py` | `_credits_table()` is imported from `commands/about/credits.py` to avoid duplication. `_wizard_export()` imports `CODEC_PROFILES`, `AUDIO_FFMPEG`, `CODEC_ALIASES`, `PRORES_CODECS`, `resolve_gpu` from `core/codec/codec_utils.py` - single source of truth for codec mappings. |
| `ui.py` | `err` console (stderr) used for all interactive/wizard output. `console` (stdout) for command results. Do not mix them. `ok()`/`warn()`/`fail()` use ASCII-safe marker `>` - Python `●`/`→` crash on CP1252 Windows terminals. |
| `core/similarity/similarity.py` | `find_similar_pairs()` accepts both `scene_index` and `index` keys for V1 (collect_scenes) / V2 (Scene.to_dict()) compat. |
| `commands/export/export.py` | `CODEC_PROFILES`/`AUDIO_FFMPEG`/`CODEC_ALIASES`/`PRORES_CODECS`/`_resolve_gpu` imported from `core/codec/codec_utils.py`. `_build_jobs()` always prefers the manifest's `[start_sec, end_sec)` range over a pre-cut clip file, falling back to the clip only when no range is given (a Scenepack's materialized clip). Keeps export mode independent of how the preview was cut: `codec="copy"` keyframe-snaps that range fresh, any other codec re-encodes it exactly - either way a `copy`-mode preview's own bleed (see `cutting.smart_cut`) never bleeds into the export. |
| `commands/sidecar/backend.py` cutting | Imports have no user-selected cut mode. `use_keyframe` scenes use `run_ffmpeg_segment_streaming()` for fast stream copy because their boundaries are keyframes. TransNetV2 scenes always use `cut_all_scenes(mode="reencode")` so previews begin at the detected frame. |
| `core/cutting/segmenter.py` | The stream-copy-then-re-encode fallback (taken for sources MP4 cannot hold, e.g. ProRes) passes `-force_key_frames` at the same epsilon-adjusted cut times as `-segment_times`: the segment muxer only splits on keyframes, so without them the fallback wrote the whole video as one segment. `run_ffmpeg_segment_streaming()` = same ffmpeg args/output as `run_ffmpeg_segment()` plus `on_segment(index, path)` / `on_progress(fraction)` callbacks. `_output_args()` skips the segment muxer entirely when `cut_points` is empty (single-GOP source, or a detector that found nothing) and writes one plain file at `output_pattern % start_num` instead - `-f segment -segment_times ""` is invalid and ffmpeg rejects it outright, which used to crash `detect_scenes` on any such source. A stream-copy split at an open-GOP keyframe loses frames: the segment muxer keeps every packet, but a segment that starts on an HEVC CRA with RASL pictures (which reference the previous GOP) has decoders discard them, so those frames appear in neither clip. Verified by packet counts (all 192 present) vs decoded frames (3 short per split) and NAL types (CRA, then RASL). Only some keyframes have them: x265 (open GOP by default, so every keyframe after the first is a CRA, not an IDR) ends the mini-GOP cleanly at a scene change and gives leading pictures only to keyframes it places by interval mid-shot. A real 24-minute x265 episode had 1 IDR, 320 CRAs, and 13 with RASL, all of them interval keyframes. So keyframe detection skips those (`generate_keyframes(skip_open_gop=True)`, from `keyframe_align.get_open_gop_keyframes`) and previews stay stream copies; `_cuts_on_open_gop_keyframes()` re-encodes only when a cut list still splits at one. The `h265_23976_cuts_opengop_mkv` fixture (`cuts_gop` layout: clean CRAs forced at cuts plus RASL-bearing interval CRAs) covers both; `MediaSpec.leading_keyframe_frames` models which keyframes have them. |
| `core/infra/ffmpeg_bootstrap.py` | Downloads a PINNED release-branch build (`ffmpeg-n8.1-latest-win64-gpl-8.1.zip`), never `ffmpeg-master-latest`. The master nightly (observed on N-125515, 2026-07-10) has a segment-muxer regression: on long multi-segment runs with re-encoded audio, later segments get a single audio packet with a large negative pts, which merges into files that start at a nonzero timestamp, mute instantly, and crash players on seek. Release 8.x builds are unaffected. Existing installs are not auto-refreshed (`is_portable_ffmpeg_installed` only checks existence) - a machine that already fetched the nightly must delete `%APPDATA%/com.amverge.cli/ffmpeg` and re-run so it re-downloads the pinned build. |
| `core/cutting/smart_cut.py` | Two cut modes, one per job (`cut_all_scenes(mode=...)`), never a per-scene decision - an earlier version guessed per scene (copy when cheap, re-encode/trim otherwise) and kept surfacing new correctness edge cases, replaced entirely. `copy`: every copy-mode cut, single-scene (`cut_scene`) or export (`engine._smartcut_ranges`), goes through `copy_range()` (frame-grid mapping, keyframe snap, remux) - do not add a second copy path; `TestCutAndExportParity` fails if the two drift apart. `snap_range_to_keyframes()` widens `[start, end)` outward to the enclosing keyframes (a no-op when already aligned) then `_lossless_copy()` remuxes that span with PyAV (not `ffmpeg -t`, which compares each packet's dts against the limit and so leaked the end keyframe plus a few reordered frames of the next scene, with holes, on every B-frame source) - always a true stream copy, no silent fallback to re-encode on failure. It stops at the end keyframe in file order, which is exact on closed GOP; on open GOP it keeps the end keyframe and its leading pictures (they display before it but decode from it), so the clip ends with exactly one frame of the next scene rather than losing the last frames of its own. Leading pictures of the start keyframe are dropped. Stream metadata/dispositions are copied; audio packets overlapping the span (including the AAC priming packet) are kept. Needs PyAV >= 14 (`add_stream_from_template`). Copy clips are `.mp4` unless `copy_container_suffix()` finds a stream the MP4 muxer rejects (asks PyAV's muxer, no codec list), then `.mov` - applies to `cut_scene` copy clips and `engine._smartcut_ranges` temp files, so ProRes copy exports work with `--container mov`. Trade-off: a clip can carry up to one keyframe interval of the neighboring scene at either edge. `reencode`: exact boundary, real encode cost. `cut_scene()` returns `poster_offset_sec` (how far into the clip the true scene start sits, nonzero only for `copy`) so thumbnail generation can seek past the bleed instead of grabbing the previous scene's frame. No mp4/mov edit list involved anywhere in this module (an earlier design used one to hide the same widening this now leaves visible - retired along with `core/cutting/editlist.py`, since hiding it turned out to freeze older After Effects builds that read the edit list literally). Every remuxed stream goes through `_add_copy_stream`: `opaque=True` (without it AV1, whose PyAV codec is the decoder-only `libdav1d`, raises `UnknownCodecError`), the source codec tag kept (else MOV relabels ProRes 4444 as HQ `apch`), and HEVC forced to `hvc1` (MKV sources carry no tag, and MP4's default `hev1` won't open in QuickTime). |
| `core/export/engine.py` | `_smartcut_ranges()` cuts range-based `ExportJob`s (`seek_ms`/`dur_ms` set) through `smart_cut.copy_range` before `_merge`/`_individual` ever run, whenever `settings.codec == "copy"`. `dedupe=True` (set for `--merge` jobs; `export.py`'s `_build_jobs()` always builds source-range jobs when the manifest has a range, so every merge sees them) groups contiguous same-source ranges and cuts each group as ONE continuous span instead of independently snapping every scene - otherwise two adjacent AI scenes whose shared cut isn't on a keyframe would each widen outward across it, duplicating that span once concatenated. Re-encode exports probe and pass through the source's `avg_frame_rate` rational (for example `24000/1001`); do not leave FFmpeg to select a container or encoder default frame rate. `ExportJob` keeps integer `seek_ms`/`dur_ms`, which miss frame boundaries (every NTSC frame, 2 of 3 frames at 24 fps), so ranges are mapped back onto the input's frame grid first (`params.frame_grid_range`, exact rate from `probe_utils.probe_video_rate`; off-grid/VFR ranges keep the old time-based path). Re-encodes then cut with `params.frame_cut_args`: seek a quarter frame early (half a frame rounds onto the previous frame on AVI, whose time base is one frame), `-frames:v` for video length (never `-t`, whose cutoff sits on a frame edge), `atrim` start/duration for audio, and `-t` only as a half-frame-padded cap for copied audio. `-frames:v` is only exact with `-fps_mode cfr` at the source rate. Copy ranges are grid-snapped the same way before `snap_range_to_keyframes`, which must be given the source `rate` so it matches keyframes within half a frame: many real files (MKV remuxes especially) store pts on a 1/1000 timescale, so a 24000/1001 keyframe sits up to 0.5 ms off the ideal grid time, and the old 1 us tolerance snapped a scene starting exactly on a keyframe back a whole GOP. The `h264_23976_ms_pts` fixture (`track_timescale=1000`) covers this; fixtures with an ideal timescale never exercise it. `smart_cut._encode_segment` (TransNetV2 previews) uses the same frame cut. Every input `-ss` is shifted by `probe_utils.probe_seek_origin` (video start minus file start, from ffprobe): ffmpeg seeks from the file start, and MKV counts AAC priming as a -21 ms audio start where MP4 hides it in an edit list, so unshifted cuts of MKV sources began one frame early. PyAV's `start_time` misses that priming and packet pts can't tell MKV priming from MP4's, which is why this probe uses ffprobe. A `copy` export never turns into a re-encode because the copy failed: `check_copy_container()` rejects, before any cutting, a container that cannot hold the source's video (ProRes into MP4) or its copied audio (PCM into MP4), and the copy merge no longer falls back to re-encoding when the concat fails. The only deliberate re-encode left in a copy merge is `_copy_concat_safe()` rejecting pre-cut clips (not source ranges) that don't start on a keyframe at 0. `check_copy_container()` raises `CopyContainerError` (structured `details`, printed as the `--ipc` error object): video is test-copied with ffmpeg and read back, since ffmpeg writes the final file and happily writes codecs a container has no mapping for; audio follows `params.audio_copy_safe`, the same table the CLI's audio fallback uses (ffmpeg writes PCM into MP4, but players don't read it). The temp clip's own container comes from `smart_cut.copy_container_suffix()` (MP4, else the source's AVI/Matroska, else MOV, each confirmed by writing a header; MP4 rejects PCM only at the header). The copy merge's concat applies the chosen audio codec (it used to always copy). AVI is copy-only (`codec_container_compatible` rejects every encoder profile; HEVC re-encoded into AVI reads back as `rawvideo`). Copies into AVI get `params.copy_video_args()`: `-r <source rate>` (from an MP4/MOV time base the AVI muxer otherwise declares 600 fps and pads with empty frames) and `H264`/`HEVC` FourCCs instead of MP4's `avc1`/`hvc1`. `_lossless_copy` starts AVI audio at the keyframe rather than keeping the slightly negative straddling packet: the muxer's shift past it rounds AVI's one-frame video time base a whole slot and repeats the first frame. `TestAviCopyExport` checks frame numbers, rate and `nb_frames` (empty frames). |
| `pipeline.py` | `DetectionMethod` is `Literal["keyframe", "edge", "transnetv2"]`. `DecodeMethod` is `Literal["ffmpeg", "nelux"]` (transnetv2 only; default `ffmpeg`). TransNetV2 path: `ffmpeg` uses `decode_and_detect_scenes()`, `nelux` uses `decode_video_frames_nelux()` + `run_model_one_pass()`, then always `cut_all_scenes(mode="reencode")` for exact previews. `nelux` runs `nelux_available()` smoke test first and falls back to `ffmpeg` if missing. Monkey-patches `emit_progress` on `ai_scene_detection`/`smart_cut` module-local refs (not `ipc` module) to route IPC progress to Rich callback. |
| `core/infra/ipc.py` | IPC protocol for Tauri app. V2 events: `PROGRESS\|pct\|msg`, `INITIAL_CLIPS_READY\|json`, `CLIP_READY\|idx\|path\|mode`, `PHASE1_COMPLETE`, `REENCODE_PROGRESS\|done\|total`, `PREVIEW_FRAME\|tag\|path\|seq`. stdout reserved for final JSON. Never mix IPC output with Rich output. |
| `--ipc` flag (hidden) | `depth-map`, `deadframes`, `interpolate` accept `--ipc`: bypasses the Rich `SystemMonitor`/`Live` display and instead streams `PROGRESS\|`/`PREVIEW_FRAME\|` on stderr (via `core/infra/preview.ipc_callbacks`), auto-confirms downloads. Standalone (non-`--ipc`) behavior is unchanged. The engines (`generate_depth_map`, `run_deadframes`, `interpolate_video`) take an additive `preview_cb(frame_bgr, pct)` param, default `None`. Used by the app to run post-export passes with a live progress+preview modal. |
| `core/detection/ai_scene_detection.py` | TransNetV2 `predictions_to_scenes` end frames are INCLUSIVE; seconds come from `scene_utils.transnet_scenes_to_seconds()` (end = `(end_frame + 1) / fps`, 6 dp), so scenes are contiguous and `scenes_frames` keeps TransNetV2's inclusive convention. `scenes_frames_to_seconds()` (2 dp, no +1) is kept for API compatibility only - do not use it for TransNetV2. The backend recomputes seconds from cached `_transnet_frames.npy` on a cache hit, so caches written before this fix are repaired. TransNetV2 inference. Requires `[ml]` extra. `TRANSNET_AVAILABLE` flag guards import at module level - raises clear `ImportError` if missing. Do not import torch at module level in other files. |
| `core/codec/codec_utils.py` | `check_if_hevc()` via ffprobe. Also contains `CODEC_PROFILES` (14 codec -> ffmpeg encoder mappings), `AUDIO_FFMPEG` (10 audio choices), `CODEC_ALIASES`, `PRORES_CODECS`, `resolve_gpu()`, `is_hevc()`, `VALID_CONTAINERS` (`avi`, `mp4`, `mov`). Single source of truth - `commands/export/export.py` and `wizard.py` import from here. Do not duplicate these dicts. |
| `core/detection/nelux_runtime.py` | Windows DLL setup for Nelux video reader. Set `AMVERGE_FFMPEG_BIN` env var to FFmpeg shared DLL directory. Idempotent - safe to call multiple times. `nelux_available()` is the quick smoke test (tries `_get_nelux_video_reader()`, returns bool, no decode) used by `detect`/`pipeline` to decide the transnetv2 decode backend. |
| `core/keyframes/keyframe_align.py` | `get_keyframe_timestamps_pyav` must never set `stream.discard` (any value) - doing so corrupts the PTS PyAV reports for the packets it does keep, off by whole frames on B-frame sources. Demux every packet and filter by `packet.is_keyframe` instead; no decode happens either way, so there is no real perf cost. Preserve packet timestamp precision: the copy export snapper passes these values directly to FFmpeg. |
| `core/thumbnails/thumbnails_streaming.py` | V1 backend mode only. Emits events as each thumbnail completes. Not used in V2 backend. |
| `core/discord/discord_rpc.py` | Uses same CLIENT_ID as AMVerge app (`1497922104065134823`). Silently no-ops if pypresence not installed. `--no-rpc` flag on detect/export/merge to disable. Methods: idle/detecting/selecting/navigating/exporting/merging/complete/error. |
| `core/upscaling/engine.py` | `upscale_model(key, ...)` - unified dispatch. Reads model from registry, routes ml → `_upscale_ml()` (spandrel, in-file), shader → `anime4k.upscale_video_anime4k()`, onnx → `artcnn.upscale_video_artcnn()`. Engine owns the ml path only; shader/onnx live in their own modules. |
| `core/upscaling/registry.json` | Declarative model registry. To add a model, add one JSON entry with method, name, scales, credit, description, file/hash. CLI auto-discovers everything. `_source` holds the ml/anime4k/artcnn base URLs; `registry.py` builds per-model `url` (ml/onnx) or `download_url` (shader zip). |
| `core/upscaling/ffmpeg_helpers.py` | Shared FFmpeg utilities used by engine/anime4k/artcnn (avoids circular import): `mux_audio()` (copies source audio with `-c:a copy`, AAC re-encode only as fallback), `build_ffmpeg_pipe()` (rawvideo stdin pipe), `get_color_args()` (probe + pass through source color primaries/transfer/matrix/range), `get_video_dims_ffprobe()`, `encode_thread_count()`, `ensure_ffmpeg()`, `CREATE_NO_WINDOW`. All encoders use `-profile:v high` with NO explicit `-level` (x264 auto-selects; hardcoding 5.1 produced non-compliant streams for 4x-of-HD outputs). |
| `core/upscaling/weight_loader.py` | Downloads ml `.pth` weights to `models/upscale/<key>/<file>`. Resume support (HTTP Range), SHA-256 integrity verification, 3 retries. ml-only; ONNX downloads live in `artcnn.py`. |
| `core/upscaling/anime4k.py` | `upscale_video_anime4k()` - REAL Anime4K. Downloads v4.0.1 GLSL shaders (`models/upscale/anime4k/`), concatenates the mode's shader chain into `_chain_<mode>_<scale>x.glsl`, applies via FFmpeg `libplacebo=custom_shader_path`. Auto-detects libplacebo (`ffmpeg -filters`); falls back to lanczos+unsharp if absent. Modes light/medium/strong = Upscale_CNN_x2_{S,M,VL} (+ Restore_CNN for medium/strong); scale=4 adds a 2nd upscale pass. **Two FFmpeg gotchas (do not reintroduce):** (1) NEVER pass global `-init_hw_device vulkan` - it breaks libplacebo output negotiation (EINVAL); libplacebo self-inits its Vulkan device. (2) `custom_shader_path` cannot take an absolute Windows path (drive colon = filtergraph option separator, no escaping survives). The chain is staged into the OUTPUT file's dir, ffmpeg runs with `cwd=that dir`, path passed as bare basename, file deleted after. |
| `core/upscaling/artcnn.py` | `upscale_video_artcnn()` - ArtCNN ONNX (luma-only 2x doublers). Downloads to `models/upscale/artcnn/<file>` (single dir - was a path-mismatch bug vs weight_loader's per-key dir). Per-frame: BGR→YUV, run Y `[1,1,H,W]` → `[1,1,2H,2W]`, lanczos-upscale U/V, recombine, rawvideo pipe. Session uses `enable_cpu_mem_arena=False` + per-frame `del`/`gc` (HD full-frame inference OOMs otherwise). **Chroma models** (entry has `chroma_file`, e.g. `R8F64_Chroma`): loads a 2nd session; chroma net takes `[1,3,H,W]` (Y_2x + bilinear-upscaled U/V) → `[1,2,H,W]` reconstructed U/V, replacing the lanczos chroma path. `download_artcnn` fetches all `_model_files(entry)`; `is_artcnn_downloaded` checks all. Input/output tensor names auto-detected. v1.6.2 assets verified. Credit: ArtCNN by Artoriuz. |
| `core/interpolation/flowframes.py` | Flowframes 1.42.0 integration. Spawns external `Flowframes.exe` with `-a -nc -mdc` args. Strips `NoDefaultCurrentDirectoryInExePath` from child env (Flowframes' bare-name ffprobe fails otherwise). Kills existing instance before spawn. Tails `FlowframesData/logs/<session>/sessionlog.txt` for progress (`Interpolated X/Y Frames`, `%`). Locates output by newest media file in `-o` dir with size stability check. |
| `commands/sidecar/rpc_server.py` | Hidden sidecar: `amverge rpc-server`. Long-lived process; Rust spawns it once and sends JSON commands via stdin (`{"type":"update","details":"...","state":"..."}`, `{"type":"clear"}`, `{"type":"shutdown"}`). Throttles Discord updates to max 1 per 15s. Exits when stdin closes or parent dies. |
| `core/interpolation/engine.py` | `interpolate_video()` - RIFE PyTorch CUDA/CPU inference. Pad frames to mod-32, cache encoded features per pair, loop `factor-1` inter-steps with saved feature restore (IFNet.forward overwrites `self.f0/f1`). FFmpeg rawvideo stdin pipe for output. Muxes source audio. Never import torch at module level outside engine/arch files. |
| `core/interpolation/rife_arch.py` | RIFEModel + IFNet architecture. Light/heavy determines IFBlock channel widths (m=1 vs m=2). `cachePair()` encodes img pair for reuse across inter-steps. Grid cache (`_tenGrid`, `_tenFlowDiv`) keyed by device/size/dtype. Parameter remapping in weight_loader handles `module.` prefix mismatches. |
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
