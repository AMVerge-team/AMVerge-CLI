# Python Library

AMVerge CLI is fully usable as a Python library. The high-level API covers most use cases; low-level modules are available for custom pipelines.

---

## Quick Start

```python
from amverge import detect_scenes

result = detect_scenes("episode.mp4")

for scene in result.scenes:
    print(scene.index, scene.start, scene.end, scene.path)

for a, b in result.similar_pairs:
    print(f"Scenes {a} and {b} look similar - consider merging")
```

---

## `detect_scenes()`

```python
from amverge import detect_scenes, DetectResult

result: DetectResult = detect_scenes(
    video_path="episode.mp4",
    output_dir="./scenes",         # defaults to <name>_scenes/ next to video
    method="keyframe",              # "keyframe" (fast) or "edge" (accurate, needs OpenCV)
    min_duration=0.25,              # merge scenes shorter than N seconds
    thumbnails=True,                # generate JPEG thumbnails
    similarity=True,                # flag adjacent scenes that look similar
    similarity_threshold=0.10,      # lower = stricter
    thumbnail_workers=4,
    edge_threshold=0.15,
    edge_radius=0.6,
    progress=lambda stage, pct, msg: print(f"[{stage}] {pct}% {msg}"),
)
```

---

## Return Types

```python
from dataclasses import dataclass

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
    scenes_json: str      # path to the saved scenes.json file
```

---

## Progress Callback

Pass a callable to `progress` to receive live updates:

```python
def on_progress(stage: str, pct: int, msg: str) -> None:
    # stage: "detect" | "segment" | "thumbnails" | "similarity"
    # pct: 0-100
    print(f"[{stage}] {pct}%  {msg}")

result = detect_scenes("episode.mp4", progress=on_progress)
```

---

## Low-Level Modules

### Binary resolution

```python
from amverge.core.binaries import get_ffmpeg, get_ffprobe

ffmpeg_path  = get_ffmpeg()   # raises if not found
ffprobe_path = get_ffprobe()
```

### Keyframe extraction

```python
from amverge.core.keyframes import generate_keyframes

timestamps: list[float] = generate_keyframes("video.mp4")
```

Uses PyAV packet demux (fast path) with a decode fallback for pathological encodes.

### Video metadata

```python
from amverge.core.video import get_video_duration, get_video_info

duration: float = get_video_duration("video.mp4")
info: dict = get_video_info("video.mp4")
# info["duration"], info["streams"] -> list of video/audio stream dicts
```

### Scene segmentation

```python
from amverge.core.segmenter import run_ffmpeg_segment, collect_scenes

run_ffmpeg_segment(video, cut_points, output_dir)
scenes = collect_scenes(output_dir, video_stem)
```

Handles Windows 32,767-char command line limit via 1500-cut chunking.

### Thumbnails

```python
from amverge.core.thumbnails import make_thumbnail, generate_thumbnails

make_thumbnail(clip_path, output_path)
generate_thumbnails(scenes, workers=4)
```

### Similarity

```python
from amverge.core.similarity import check_pair_similar, find_similar_pairs

similar: bool = check_pair_similar(thumb_a, thumb_b, threshold=0.10)
pairs: list[tuple[int, int]] = find_similar_pairs(scenes, threshold=0.10)
```

Uses cosine similarity on average-pooled pixel arrays.

### HEVC detection

```python
from amverge.core.hevc import is_hevc

is_hevc("video.mp4")  # True / False
```

### Image crop

```python
from amverge.core.image import crop_image, CropData

crop_image(
    "input.jpg",
    "output.jpg",
    CropData(x=10, y=10, width=200, height=200, rotation=90),
)
```

Supports animated GIF input/output.

### Detection methods (direct access)

```python
from amverge.core.detection import detect_cuts_by_keyframe, detect_cuts_by_edge

cut_points: list[float] = detect_cuts_by_keyframe("video.mp4", min_duration=0.25)
cut_points: list[float] = detect_cuts_by_edge("video.mp4", threshold=0.15)  # needs [edge]
```
