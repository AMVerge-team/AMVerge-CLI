# Scene Cutting

Low-level scene cutting using smart cut (lossless copy / smartcut / re-encode)
and FFmpeg segment muxer.

---

## Examples

| file | description |
|---|---|
| [01_smart_cut.py](01_smart_cut.py) | V2 pipeline: smart cut with auto mode selection |
| [02_ffmpeg_segment.py](02_ffmpeg_segment.py) | V1 pipeline: FFmpeg segment muxer |
| [03_single_scene.py](03_single_scene.py) | cut one scene, inspect mode |

---

## Quick Run

```bash
python examples/cutting/01_smart_cut.py episode.mp4
```
