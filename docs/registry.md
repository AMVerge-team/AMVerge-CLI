# Registry Format

`amverge/core/upscaling/registry.json` is the single source of truth for all upscale models. Adding a model is one JSON entry. CLI auto-discovers everything.

## Structure

```json
{
  "_source": { ... },
  "presets": { ... },
  "models": { ... }
}
```

### `_source`

Base URLs for each method type. Models derive their download URL from these.

```json
"_source": {
  "ml": "https://github.com/user/repo/releases/download/",
  "anime4k": "https://github.com/bloc97/Anime4K/...",
  "artcnn": "https://github.com/Artoriuz/ArtCNN/releases/download/v1.6.2/"
}
```

| key | used by | how URL is built |
|-----|---------|------------------|
| `ml` | method `ml` | `{ml}{category}/{file}` |
| `artcnn` | method `onnx` | `{artcnn}{file}` |
| `anime4k` | method `shader` | used directly (zip download) |

### `presets`

Encoding quality presets. All presets use `tune=animation`. Changing these affects ALL upscale methods.

```json
"presets": {
  "archival": {"crf": 14, "x264": "slow",      "tune": "animation"},
  "high":     {"crf": 17, "x264": "slow",      "tune": "animation"},
  "balanced": {"crf": 20, "x264": "medium",    "tune": "animation"},
  "fast":     {"crf": 22, "x264": "veryfast",  "tune": "animation"},
  "draft":    {"crf": 26, "x264": "ultrafast", "tune": "animation"}
}
```

| field | meaning | typical range |
|-------|---------|---------------|
| `crf` | H.264 constant rate factor (lower = better) | 14 (archival) to 26 (draft) |
| `x264` | x264 speed preset | ultrafast, veryfast, fast, medium, slow |
| `tune` | x264 tune parameter | always `animation` for anime content |

### `models`

Each key becomes a CLI model name. Available flags depend on `method`.

---

## Method Types

### `method: "ml"`

PyTorch model loaded via spandrel. Frame-by-frame inference on GPU/CPU.

| field | required | description |
|-------|----------|-------------|
| `method` | yes | `"ml"` |
| `name` | yes | display name in lists and prompts |
| `scales` | yes | `[2]` or `[2, 4]` - supported scale factors |
| `credit` | yes | attribution shown in `--credits` |
| `description` | yes | short text shown in `--list-models` |
| `category` | yes | subfolder on the download server (usually `"upscale"`) |
| `file` | yes | the `.pth` filename |
| `hash` | yes | SHA-256 hex digest for integrity verification |

```json
"adore": {
  "method": "ml",
  "name": "Adore",
  "scales": [2, 4],
  "credit": "based on AniSmooth by moongetsu",
  "description": "High quality anime upscaler",
  "category": "upscale",
  "file": "adore.pth",
  "hash": "443378bdc6db6cf4a75eea61ee7afc78b2c4b6a4d3b3981a40ff61f38bbc8f1a"
}
```

Download URL becomes: `{_source.ml}{category}/{file}` = `AniSmooth-Models/.../upscale/adore.pth`

Model files are stored at `%APPDATA%/com.amverge.cli/models/upscale/{key}/{file}`.

---

### `method: "shader"`

FFmpeg filter pipeline. No ML deps, very fast. Uses lanczos upscale + unsharp mask + smartblur.

| field | required | description |
|-------|----------|-------------|
| `method` | yes | `"shader"` |
| `name` | yes | display name |
| `scales` | yes | supported scaling factors |
| `credit` | yes | attribution |
| `description` | yes | short text |
| `modes` | no | list of intensity presets (default: `["medium"]`) |
| `default_mode` | no | which mode to use if none specified |

```json
"anime4k": {
  "method": "shader",
  "name": "Anime4K",
  "scales": [2, 4],
  "credit": "by bloc97 (MIT License)",
  "description": "GPU shader-based upscaler, no ML deps",
  "modes": ["light", "medium", "strong"],
  "default_mode": "medium"
}
```

Mode effects:
- `light`: lanczos upscale only
- `medium`: lanczos + unsharp(5:5:0.8) + smartblur(0.8)
- `strong`: lanczos + unsharp(7:7:1.0) + smartblur(1.0)

The pipeline is defined in `core/upscaling/engine.py` `_upscale_shader()`. Modes control filter intensity.

No shader files are needed - the pipeline uses FFmpeg built-in filters.

---

### `method: "onnx"`

ONNX Runtime inference. Lighter than PyTorch, no spandrel needed. Luma-only (Y channel) processing.

| field | required | description |
|-------|----------|-------------|
| `method` | yes | `"onnx"` |
| `name` | yes | display name |
| `scales` | yes | currently `[2]` only (model-specific) |
| `credit` | yes | attribution |
| `description` | yes | short text |
| `file` | yes | the `.onnx` filename |
| `input_channels` | no | channels expected by model (default: 1 for luma-only) |

```json
"C4F32": {
  "method": "onnx",
  "name": "ArtCNN C4F32",
  "scales": [2],
  "credit": "by Artoriuz",
  "description": "Balanced real-time anime upscaler (48K params)",
  "file": "ArtCNN_C4F32.onnx",
  "input_channels": 1
}
```

Download URL becomes: `{_source.artcnn}{file}` = `ArtCNN/.../v1.6.2/ArtCNN_C4F32.onnx`

If `input_channels` is 1 (default), the pipeline converts BGR to YUV, runs the model on Y channel only, upscales UV with lanczos, then recombines. All handled in `_upscale_onnx()`.

ONNX files are stored at `%APPDATA%/com.amverge.cli/models/upscale/artcnn/`.

---

## After Adding a Model

The model appears in these commands automatically:

```
amverge upscale --list-models     model listed with name, method, description
amverge upscale --credits         credit added (deduplicated)
amverge upscale episode.mp4 -m <key>    dispatched to correct pipeline automatically
amverge models                    listed in appropriate category table
amverge models --download <key>   downloads to correct path
amverge models --delete <key>     deletes from disk
```

No Python changes needed. No CLI changes needed. No import changes needed.

## Engine Dispatch

`core/upscaling/engine.py` is the single file that handles all method types. `upscale_model(key, ...)` reads the registry entry and dispatches:

| `method` value | pipeline function | description |
|----------------|-------------------|-------------|
| `ml` | `_upscale_ml()` | OpenCV frame read → spandrel model → FFmpeg pipe |
| `shader` | `_upscale_shader()` | FFmpeg lanczos + unsharp + smartblur filter chain |
| `onnx` | `_upscale_onnx()` | OpenCV → YUV split → ONNX session → FFmpeg pipe |

The CLI just calls `upscale_model()` with the model key. The method is determined from the registry entry.
