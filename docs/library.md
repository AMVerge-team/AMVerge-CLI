# Python Library

AMVerge CLI is fully usable as a Python library. Everything is importable from
the top-level ``amverge`` package - no deep ``core.*`` paths needed.

```python
import amverge

# Detect scenes
result = amverge.detect_scenes("episode.mp4")

# Export a single clip
amverge.make_thumbnail("scene_0001.mp4", "thumb.jpg")

# Check codec
is_hevc = amverge.check_if_hevc("episode.mp4")
```

---

## CLI → Python Command Map

Every CLI command has a Python library equivalent.

### `amverge detect`

```python
import amverge

result = amverge.detect_scenes("episode.mp4")
for scene in result.scenes:
    print(f"Scene {scene.index}: {scene.start:.1f}s - {scene.end:.1f}s")
```

### `amverge export`

```python
import amverge
import json
from pathlib import Path

# Read scenes.json
data = json.loads(Path("scenes.json").read_text())
scenes = data["scenes"]

# Copy mode (lossless)
for s in scenes:
    amverge.make_thumbnail(s["path"], f"thumb_{s['index']:04d}.jpg")
```

### `amverge merge`

```python
import subprocess
from amverge import get_ffmpeg

subprocess.run([
    get_ffmpeg(), "-y",
    "-f", "concat", "-safe", "0",
    "-i", "concat.txt",   # file 'clip1.mp4'\nfile 'clip2.mp4'
    "-c", "copy",
    "merged.mp4",
], check=True)
```

### `amverge info`

```python
from amverge import get_video_info, get_video_duration

info = get_video_info("episode.mp4")
print(f"Duration: {get_video_duration('episode.mp4'):.1f}s")
for s in info["streams"]:
    if s["type"] == "video":
        print(f"Video: {s['codec']} {s['width']}x{s['height']} {s['fps']}fps")
    elif s["type"] == "audio":
        print(f"Audio: {s['codec']} {s['sample_rate']}Hz {s['channels']}ch")
```

### `amverge probe`

```python
from amverge import (
    probe_video_duration, probe_video_fps, probe_video_dimensions,
    check_if_hevc, get_keyframe_timestamps_pyav,
)

fps = probe_video_fps("episode.mp4")
w, h = probe_video_dimensions("episode.mp4")
dur = probe_video_duration("episode.mp4")
hevc = check_if_hevc("episode.mp4")
kf = get_keyframe_timestamps_pyav("episode.mp4")

print(f"{w}x{h} {fps}fps {dur:.1f}s  HEVC={hevc}  {len(kf)} keyframes")
```

### `amverge gpu`

```python
import torch

print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

### `amverge keyframes`

```python
from amverge import get_keyframe_timestamps_pyav, generate_keyframes

# V2 (TransNetV2 pipeline)
kf = get_keyframe_timestamps_pyav("episode.mp4")

# V1 (with progress callback)
kf = generate_keyframes("episode.mp4", progress_cb=lambda pct, msg: print(f"{pct}%"))
```

### `amverge scenes`

```python
import numpy as np

scenes_secs = np.load("scenes_be84f8c8a759_secs.npy")
for start, end in scenes_secs:
    print(f"Scene: {start:.1f}s - {end:.1f}s")
```

### `amverge cache`

```python
from amverge import build_video_cache_prefix
from pathlib import Path

prefix = build_video_cache_prefix(Path("episode.mp4"))
secs_path = Path("scenes") / f"{prefix}_secs.npy"

if secs_path.exists():
    secs_path.unlink()
    print("Cache cleared")
```

### `amverge models`

```python
from amverge import UPSCALE_REGISTRY, get_ml_models, get_onnx_models, get_shader_models

# Query the registry (loaded from registry.json)
print("All models:", list(UPSCALE_REGISTRY.keys()))
print("ML models:", list(get_ml_models().keys()))
print("ONNX models:", list(get_onnx_models().keys()))

# Model metadata
for key, entry in UPSCALE_REGISTRY.items():
    print(f"{key}: {entry['name']} ({entry['method']}) {entry['scales']}")
    print(f"  {entry.get('description', '')}")

from amverge import MODEL_FILES, UPSCALE_MODEL_KEYS, is_weight_downloaded

print("Available ML models:", UPSCALE_MODEL_KEYS)
for key in UPSCALE_MODEL_KEYS:
    print(f"  {key}: {'downloaded' if is_weight_downloaded(key) else 'not downloaded'}")
```

---

### `amverge upscale`

```python
from amverge import upscale_video, QUALITY_PRESETS

# ML method (needs torch + opencv)
upscale_video(
    "episode.mp4", "upscaled.mp4",
    model_name="adore", scale=2, preset="high",
    progress_cb=lambda pct, msg: print(f"{pct}% {msg}"),
)

# Anime4K shader method (FFmpeg only, fast)
from amverge import upscale_video_anime4k

upscale_video_anime4k(
    "episode.mp4", "upscaled.mp4",
    scale=2, mode="medium", preset="high",
)

# ArtCNN ONNX method (lightweight, fast)
from amverge import upscale_video_artcnn

upscale_video_artcnn(
    "episode.mp4", "upscaled.mp4",
    model_name="C4F32", scale=2, preset="high",
)
```

---

### `amverge version`

```python
import amverge

print(amverge.__version__)
```

---

## `detect_scenes()` Reference

```python
from amverge import detect_scenes

# Keyframe (default, no deps)
result = detect_scenes("episode.mp4")

# Edge detection (needs pip install amverge[edge])
result = detect_scenes("episode.mp4", method="edge")

# TransNetV2 ML (needs pip install amverge[ml])
result = detect_scenes("episode.mp4", method="transnetv2")

# Full control
result = detect_scenes(
    video_path="episode.mp4",
    output_dir="./scenes",
    method="keyframe",              # "keyframe" | "edge" | "transnetv2"
    min_duration=0.25,              # merge scenes shorter than N seconds
    thumbnails=True,                # generate JPEG thumbnails
    similarity=True,                # flag adjacent scenes that look similar
    similarity_threshold=0.10,      # lower = stricter
    thumbnail_workers=4,
    edge_threshold=0.15,            # edge method only
    edge_radius=0.6,                # edge method only
    progress=lambda stage, pct, msg: print(f"[{stage}] {pct}% {msg}"),
)
```

---

## Return Types

```python
@dataclass
class Scene:
    index: int
    start: float          # seconds from video start
    end: float            # seconds from video start
    duration: float       # seconds
    path: str             # absolute path to the .mp4 clip
    thumbnail: str | None # absolute path to the .jpg thumbnail (None if skipped)
    original_file: str    # stem of the source video

@dataclass
class DetectResult:
    scenes: list[Scene]
    similar_pairs: list[tuple[int, int]]  # (scene_index_a, scene_index_b)
    output_dir: str
    scenes_json: str
```

---

## Progress Callback

```python
from amverge import detect_scenes

def on_progress(stage: str, pct: int, msg: str) -> None:
    # stage: "detect" | "segment" | "thumbnails" | "similarity"
    print(f"[{stage}] {pct}%  {msg}")

result = detect_scenes("episode.mp4", progress=on_progress)
```

---

## Complete Module Reference

All names are available via ``from amverge import <name>``.

### Binaries

```python
from amverge import get_binary, get_ffmpeg, get_ffprobe

ffmpeg  = get_ffmpeg()   # ffmpeg binary path
ffprobe = get_ffprobe()
binary  = get_binary("ffmpeg")
```

### Video Metadata

```python
from amverge import (
    get_video_duration, get_video_info, merge_short_scenes,
    probe_video_fps, probe_video_dimensions,
    probe_video_duration, probe_video_total_frames,
)

fps     = probe_video_fps("video.mp4")
w, h    = probe_video_dimensions("video.mp4")
dur     = probe_video_duration("video.mp4")
frames  = probe_video_total_frames("video.mp4", fps, dur)
info    = get_video_info("video.mp4")
merged  = merge_short_scenes([0.0, 0.2, 5.0, 10.0], min_duration=0.5)
```

### Keyframes

```python
from amverge import (
    generate_keyframes, get_keyframe_timestamps_pyav,
    classify_scenes_by_keyframe_alignment,
)

# V1 pipeline (with progress)
kf = generate_keyframes("video.mp4")

# V2 pipeline (PyAV demux)
kf = get_keyframe_timestamps_pyav("video.mp4")

# Classify scenes for lossless copy vs re-encode
copy, reencode = classify_scenes_by_keyframe_alignment(
    [(0.0, 5.0), (5.2, 10.0)], kf
)
```

### Scene Detection V1

```python
from amverge import detect_cuts_by_keyframe, detect_cuts_by_edge

cuts = detect_cuts_by_keyframe("video.mp4", min_duration=0.25)
cuts = detect_cuts_by_edge("video.mp4", threshold=0.15)  # needs [edge]
```

### Scene Detection V2 (TransNetV2)

```python
from amverge import (
    TRANSNET_AVAILABLE, decode_and_detect_scenes,
    decode_video_frames_nelux, run_model_one_pass,
)

if not TRANSNET_AVAILABLE:
    print("Run: pip install amverge[ml]")
else:
    # One-shot: FFmpeg pipe decode + inference
    secs, frames = decode_and_detect_scenes("video.mp4")

    # Step-by-step (custom pipeline)
    frames = decode_video_frames_nelux("video.mp4")  # numpy (N, 27, 48, 3)
    secs, frames = run_model_one_pass(frames, "video.mp4")
```

### Scene Cutting

```python
from amverge import cut_scene, cut_all_scenes, run_ffmpeg_segment, collect_scenes
from pathlib import Path

# Smart cut (V2 pipeline - handles copy/smartcut/reencode)
results = cut_all_scenes(
    input_file=Path("video.mp4"),
    scenes=[{"scene_index": 0, "start_sec": 0.0, "end_sec": 5.0}],
    keyframes=[0.0, 0.5, 1.0, 1.5, 5.0],
    out_dir=Path("./scenes"),
    use_cuda=True, is_hevc=False,
    on_ready=lambda r: print(r["clip_mode"]),
)

# FFmpeg segment (V1 pipeline - lossless stream copy)
run_ffmpeg_segment("video.mp4", "out_%04d.mp4", [5.0, 10.0, 15.0])
scenes = collect_scenes("./out", "video", [5.0, 10.0, 15.0], 20.0)
```

### Scene Utilities

```python
from amverge import (
    scenes_frames_to_seconds, convert_scenes_to_timestamps, scenes_to_objects,
)
import numpy as np

secs = scenes_frames_to_seconds(np.array([[0, 120], [120, 240]]), fps=24.0)
ts, cuts = convert_scenes_to_timestamps("video.mp4", np.array([[0, 120]]))
objs = scenes_to_objects(secs, np.array([[0, 120], [120, 240]]))
```

### Thumbnails

```python
from amverge import make_thumbnail, generate_thumbnails

make_thumbnail("clip.mp4", "thumb.jpg")  # returns True if success

scenes = [{"scene_index": 0}, {"scene_index": 1}]
generate_thumbnails(scenes, "./out", "episode", workers=4)
```

### Similarity

```python
from amverge import check_pair_similar, find_similar_pairs

similar = check_pair_similar("thumb_a.jpg", "thumb_b.jpg", threshold=0.10)

scenes = [
    {"scene_index": 0, "thumbnail": "th_0000.jpg"},
    {"scene_index": 1, "thumbnail": "th_0001.jpg"},
]
pairs = find_similar_pairs(scenes)
for a, b in pairs:
    print(f"Scenes {a} and {b} look similar")
```

### Codec Detection

```python
from amverge import check_if_hevc, is_hevc

check_if_hevc("video.mp4")  # True/False via ffprobe
is_hevc("video.mp4")        # same, V1 API
```

### Image Crop

```python
from amverge import CropData, crop_image

crop = CropData(x=10, y=10, width=200, height=200, rotation=90)
crop = CropData.from_dict({"x": 10, "y": 10, "width": 200, "height": 200})
crop_image("input.jpg", "output.jpg", crop)
```

### IPC Events

```python
from amverge import emit_progress, emit_event, log

emit_progress(50, "Halfway done")
emit_event("CLIP_READY|0|path/to/clip.mp4|copy")
log("Debug message")
```

### Discord RPC

```python
from amverge import RPC_AVAILABLE, DiscordRPC

if RPC_AVAILABLE:
    rpc = DiscordRPC()
    rpc.connect()
    rpc.update_detecting("episode.mp4", percent=50)
    rpc.update_exporting("episode.mp4")
    rpc.clear_presence()
    rpc.disconnect()
```

### TransNetV2 Constants

```python
from amverge import FRAME_WIDTH, FRAME_HEIGHT, FRAME_CHANNELS, FRAME_BYTES, WINDOW_SIZE, STRIDE

print(f"TransNetV2 input: {FRAME_WIDTH}x{FRAME_HEIGHT} {FRAME_CHANNELS}ch")
print(f"Window: {WINDOW_SIZE} frames, stride: {STRIDE}")
```

### Cache Utilities

```python
from amverge import build_video_cache_prefix, check_if_path_exists
from pathlib import Path

prefix = build_video_cache_prefix(Path("episode.mp4"))
secs = Path("scenes") / f"{prefix}_secs.npy"

check_if_path_exists(str(secs))  # raises FileNotFoundError if missing
```

### Upscaling

```python
from amverge import (
    UPSCALE_AVAILABLE, QUALITY_PRESETS,
    UPSCALE_REGISTRY, get_ml_models, get_onnx_models, get_shader_models,
    upscale_video, upscale_video_anime4k, upscale_video_artcnn,
    download_weights, is_weight_downloaded, get_weight_path,
    verify_weight_hash, load_weights_if_available,
    ANIME4K_MODE_PRESETS, ANIME4K_SHADER_FILES, ARTCNN_MODELS,
)

# Check availability
print("Upscale available:", UPSCALE_AVAILABLE)

# Quality presets (from registry.json)
print("Presets:", list(QUALITY_PRESETS.keys()))
# {'archival': {'crf': 14, 'x264': 'slow', 'tune': 'animation'}, ...}

# Model registry - all models from registry.json
for key, entry in UPSCALE_REGISTRY.items():
    print(f"{key}: {entry['name']} ({entry['method']}) scales={entry['scales']}")
    print(f"  {entry.get('description')} | {entry.get('credit')}")

# Query by method
ml_models = get_ml_models()          # {"adore": {...}, "shufflecugan": {...}, ...}
onnx_models = get_onnx_models()      # {"C4F16": {...}, "C4F32": {...}, ...}
shader_models = get_shader_models()  # {"anime4k": {...}}

# Anime4K shader modes
print("Anime4K modes:", list(ANIME4K_MODE_PRESETS.keys()))  # ['light', 'medium', 'strong']

# Weight management
download_weights("adore")                  # downloads from registry URL
is_weight_downloaded("shufflecugan")       # True/False
path = get_weight_path("adore")            # full path to .pth file
verify_weight_hash("adore", path)          # SHA-256 integrity check

# ML upscale pipeline (torch required)
if UPSCALE_AVAILABLE:
    upscale_video(
        "episode.mp4", "upscaled.mp4",
        model_name="adore", scale=2, preset="high",
        fit_w=1920, fit_h=1080,
        progress_cb=lambda pct, msg: print(f"[{pct}%] {msg}"),
    )

# Anime4K shader upscale (FFmpeg only, no ML deps)
upscale_video_anime4k(
    "episode.mp4", "upscaled.mp4",
    scale=2, mode="medium", preset="high",
)

# ArtCNN ONNX upscale (needs onnxruntime)
upscale_video_artcnn(
    "episode.mp4", "upscaled.mp4",
    model_name="C4F32", scale=2, preset="high",
)
```
