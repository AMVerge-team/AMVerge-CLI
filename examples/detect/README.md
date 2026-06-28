# Scene Detection

Split a video into scenes at cut boundaries using AMVerge's detection engine.

---

## How It Works

```txt
video file
     ↓
detection method (keyframe / edge / TransNetV2)
     ↓
cut timestamps
     ↓
ffmpeg segment or smart cut
     ↓
.mp4 clips + .jpg thumbnails + scenes.json
```

---

## Examples

| file | method | deps |
|---|---|---|
| [01_basic_detect.py](01_basic_detect.py) | keyframe (default) | none |
| [02_transnetv2_detect.py](02_transnetv2_detect.py) | TransNetV2 ML | `[ml]` |
| [03_edge_detect.py](03_edge_detect.py) | Canny edge + cosine | `[edge]` |
| [04_custom_settings.py](04_custom_settings.py) | keyframe with custom params | none |

---

## Quick Run

```bash
python examples/detect/01_basic_detect.py episode.mp4
python examples/detect/02_transnetv2_detect.py episode.mp4
```

Output goes to `<video_name>_scenes/` with clips, thumbnails, and a `scenes.json` index.
