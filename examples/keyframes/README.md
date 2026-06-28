# Keyframes

Extract keyframe (I-frame) timestamps and classify scenes for lossless copy
or re-encode.

---

## Examples

| file | description |
|---|---|
| [01_extract_keyframes.py](01_extract_keyframes.py) | V1 + V2 keyframe extraction |
| [02_align_scenes.py](02_align_scenes.py) | classify scenes by keyframe alignment |

---

## Quick Run

```bash
python examples/keyframes/01_extract_keyframes.py episode.mp4
python examples/keyframes/02_align_scenes.py episode.mp4
```
