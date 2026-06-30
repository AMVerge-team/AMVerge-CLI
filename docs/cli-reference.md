# CLI Reference

Run `amverge` with no arguments to open the interactive wizard.
Run `amverge --help` to see all commands.

---

## Workflow Commands

### `amverge detect`

Split a video into scenes at cut boundaries.

```bash
amverge detect episode.mp4                              # keyframe (fast, default)
amverge detect episode.mp4 --method edge                 # edge detection (accurate, needs opencv)
amverge detect episode.mp4 --method transnetv2           # ML detection (best accuracy, needs torch)
amverge detect episode.mp4 --output ./scenes
amverge detect episode.mp4 --format json > scenes.json
amverge detect episode.mp4 --no-thumbnails --no-similarity
amverge detect episode.mp4 --min-duration 0.5 --workers 8
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output / -o` | `<name>_scenes/` | Output directory |
| `--method / -m` | `keyframe` | `keyframe`, `edge`, or `transnetv2` |
| `--format / -f` | `table` | `table`, `json`, or `paths` |
| `--json-output` | - | Also save JSON to a file |
| `--no-thumbnails` | false | Skip thumbnail generation |
| `--no-similarity` | false | Skip similarity check |
| `--min-duration` | `0.25` | Merge scenes shorter than N seconds |
| `--workers` | `4` | Thumbnail thread count |
| `--similarity-threshold` | `0.10` | Similarity cutoff (lower = stricter) |
| `--edge-threshold` | `0.15` | Edge detection sensitivity |
| `--edge-radius` | `0.6` | Keyframe window radius (edge method) |
| `--no-rpc` | false | Disable Discord RPC |

**Output:** scene clips (`.mp4`), thumbnails (`.jpg`), and a `scenes.json` index.

**Detection methods:**

| method | speed | accuracy | requires |
|--------|-------|----------|----------|
| `keyframe` | fast | good | nothing extra |
| `transnetv2` | medium | best | `pip install amverge[ml]` |
| `edge` | slow | excellent | `pip install amverge[edge]` |

See [detection-methods.md](detection-methods.md) for details.

---

### `amverge export`

Export selected scenes from a previous detect run.

```bash
amverge export episode.mp4 --scenes episode_scenes/scenes.json
amverge export episode.mp4 --scenes scenes.json --output ./export
amverge export episode.mp4 --scenes scenes.json --select 0,2,5-8
amverge export episode.mp4 --scenes scenes.json --select 0-10 --merge
amverge export episode.mp4 --scenes scenes.json --codec h264 --audio aac_320
amverge export episode.mp4 --scenes scenes.json --codec h265_main --hardware gpu
amverge export episode.mp4 --scenes scenes.json --codec prores_422 --container mov
```

| Flag | Default | Description |
|------|---------|-------------|
| `--scenes / -s` | required | `scenes.json` from `detect` |
| `--output / -o` | `./export` | Output directory |
| `--select` | all | Index range: `0,2,5-8` |
| `--merge` | false | Merge selection into one file |
| `--codec` | `copy` | Codec profile (see below) |
| `--audio` | `copy` | Audio codec (see below) |
| `--container` | `mp4` | Output container: `mp4`, `mkv`, `mov` |
| `--hardware` | `auto` | GPU encode: `auto`, `gpu`, `cpu` |
| `--no-rpc` | false | Disable Discord RPC |

**Codec profiles:**

| codec | description | GPU encoder |
|-------|-------------|-------------|
| `copy` | stream copy, no quality loss | - |
| `h264` | alias for `h264_main` | - |
| `h264_main` | H.264 main profile | h264_nvenc |
| `h264_high` | H.264 high profile | h264_nvenc |
| `h264_high10` | H.264 high 10-bit | CPU only |
| `h264_high422` | H.264 high 4:2:2 | CPU only |
| `hevc` / `h265` | alias for `h265_main` | - |
| `h265_main` | HEVC main profile | hevc_nvenc |
| `h265_main10` | HEVC main 10-bit | hevc_nvenc |
| `h265_main12` | HEVC main 12-bit | CPU only |
| `h265_main422_10` | HEVC main 4:2:2 10-bit | CPU only |
| `av1_main` | AV1 main profile | av1_nvenc |
| `prores_422_lt` | ProRes 422 LT | CPU only |
| `prores_422` | ProRes 422 | CPU only |
| `prores_422_hq` | ProRes 422 HQ | CPU only |
| `prores_4444` | ProRes 4444 | CPU only |
| `prores_4444_xq` | ProRes 4444 XQ | CPU only |

ProRes codecs require `--container mov`.

**Audio codecs:**

| audio | ffmpeg flags |
|-------|-------------|
| `copy` | `-c:a copy` |
| `aac` | `-c:a aac` |
| `aac_320` | `-c:a aac -b:a 320k` |
| `pcm16` | `-c:a pcm_s16le` |
| `pcm24` | `-c:a pcm_s24le` |
| `flac` | `-c:a flac` |
| `alac` | `-c:a alac` |
| `opus` | `-c:a libopus` |
| `mp3` | `-c:a libmp3lame` |
| `none` | `-an` |

---

### `amverge merge`

Merge multiple clips into a single file using FFmpeg concat.

```bash
amverge merge clip1.mp4 clip2.mp4 clip3.mp4 --output merged.mp4
```

| Flag | Default | Description |
|------|---------|-------------|
| `--output / -o` | required | Output file path |

---

### `amverge info`

Show video and audio stream metadata.

```bash
amverge info episode.mp4
```

Displays codec, resolution, FPS, bitrate, sample rate, and channel count.

---

## Diagnostic Commands

### `amverge probe`

V2 pipeline diagnostics: codec, HEVC check, keyframe stats, scene cache status.

```bash
amverge probe episode.mp4
amverge probe episode.mp4 --no-keyframes
amverge probe episode.mp4 --cache-dir ./scenes
```

### `amverge gpu`

PyTorch version, CUDA availability, GPU name and VRAM, optional deps status.

```bash
amverge gpu
```

### `amverge doctor`

Full environment health check: FFmpeg, all deps, temp dir write access. Pass/fail per check.

```bash
amverge doctor
```

### `amverge version`

CLI version, Python version, and all dependency versions.

```bash
amverge version
amverge version --json    # machine-readable for bug reports
```

### `amverge bench`

Benchmark keyframe scan and TransNetV2 decode/inference timing.

```bash
amverge bench episode.mp4
amverge bench episode.mp4 --skip-ml
```

---

## Utility Commands

### `amverge upscale`

Upscale video using AI super-resolution or GPU-accelerated filters.
Method is auto-detected from the model key in registry.json.

```bash
amverge upscale episode.mp4 -m adore -s 2               # ML model (GPU recommended)
amverge upscale episode.mp4 -m anime4k --mode medium    # shader (fast, no ML)
amverge upscale episode.mp4 -m C4F32 -s 2               # ONNX (lightweight)
amverge upscale episode.mp4 -m realesrgan-x2            # Real-ESRGAN x2
amverge upscale episode.mp4 -m adore -s 4 -p archival   # archive quality 4x
amverge upscale --list-models                           # browse all models
amverge upscale --credits                               # show credits
amverge upscale episode.mp4 -m adore -y                 # auto-confirm downloads
```

| Flag | Default | Description |
|------|---------|-------------|
| `--model / -m` | `adore` | Model key from registry (see --list-models) |
| `--output / -o` | `upscaled.mp4` | Output video file |
| `--scale / -s` | `2` | Scale factor (model-specific, 2 or 4) |
| `--preset / -p` | `high` | Quality: archival, high, balanced, fast, draft |
| `--mode` | `medium` | Shader intensity: light, medium, strong |
| `--fit-w` | `0` | Max output width (0 = no limit) |
| `--fit-h` | `0` | Max output height (0 = no limit) |
| `--list-models` | false | Show all available models and descriptions |
| `--credits` | false | Show credits for all upscaling technologies |
| `--yes / -y` | false | Auto-confirm all download prompts |
| `--no-monitor` | false | Disable live GPU/CPU/RAM/ETA display |

**Method dispatch** (automatic from registry.json):

| method | speed | deps | description |
|--------|-------|------|-------------|
| `shader` | fastest | FFmpeg only | Lanczos + unsharp + smartblur filters |
| `onnx` | fast | `[upscale]` | ONNX Runtime inference (ArtCNN) |
| `ml` | medium | `[upscale]` | Spandrel auto-arch detection (RealCUGAN, ESRGAN, etc.) |

**System monitor:** During ml and onnx upscaling, a live panel shows GPU utilization, VRAM, CPU%, RAM, ETA, and fps. Use `--no-monitor` to disable.

**Adding models:** Edit `amverge/core/upscaling/registry.json`. One JSON entry per model. See `docs/registry.md` for format.

Models and FFmpeg are auto-downloaded to `%APPDATA%/com.amverge.cli/`. On first run, prompts ask before downloading (skip with `--yes`).

---

### `amverge models`

Manage upscaling model files. Lists all models from the registry.

```bash
amverge models                             # list all downloaded models
amverge models --download adore            # download a model
amverge models --download anime4k          # download Anime4K shaders
amverge models --download C4F32            # download ArtCNN model
amverge models --delete shufflecugan       # delete a model from disk
amverge models --delete anime4k            # delete all Anime4K shaders
amverge models --storage                   # show cache directories
```

| Flag | Default | Description |
|------|---------|-------------|
| `--download` | - | Download a model by key |
| `--delete` | - | Delete a model by key |
| `--storage` | false | Show storage directories |

**Model keys:** Use `amverge upscale --list-models` for the full list. Keys come from `registry.json`.
Storage: `%APPDATA%/com.amverge.cli/models/upscale/{key}/`

---

### `amverge keyframes`

Dump keyframe timestamps from a video.

```bash
amverge keyframes episode.mp4
amverge keyframes episode.mp4 --json
amverge keyframes episode.mp4 --count    # keyframe count only
```

### `amverge scenes`

Show scene list from a `.npy` scene cache (created by TransNetV2 detect).

```bash
amverge scenes episode.mp4
amverge scenes episode.mp4 --json
amverge scenes episode.mp4 --min-duration 0.5
amverge scenes episode.mp4 --cache-dir ./scenes
```

### `amverge cache`

List or clear TransNetV2 `.npy` scene caches.

```bash
amverge cache ./scenes                    # list caches in directory
amverge cache ./scenes --clear episode.mp4 # clear cache for specific video
amverge cache ./scenes --clear-all         # clear all caches in directory
```

---

## Info Commands

```bash
amverge usage      # command reference and examples
amverge about      # what AMVerge CLI is
amverge credits    # contributors
amverge changelog  # version history
```

---

## Interactive Wizard

Running `amverge` with no arguments opens a full interactive session.
Each command walks through its options step by step with a summary before running.

```bash
amverge
```

---

## Piping / Automation

All wizard output goes to stderr. Command results go to stdout.
This keeps stdout clean for piping:

```bash
amverge detect episode.mp4 --format json | python process.py
amverge keyframes episode.mp4 --json | jq '.[]'
amverge version --json > bug-report.json
```
