# Custom Pipeline

Full end-to-end scene detection and cutting pipeline built from low-level
modules. Shows how to replicate `amverge detect --method transnetv2`
step by step.

---

## What It Does

```txt
video file
     ↓
TransNetV2 decode + inference  (scene_detection.py)
     ↓
scene boundaries in seconds     (scene_utils.py)
     ↓
keyframe timestamps             (keyframe_align.py)
     ↓
HEVC check                      (codec_utils.py)
     ↓
classify scenes by alignment    (keyframe_align.py)
     ↓
Phase 1: lossless copy          (smart_cut.py)
     ↓
Phase 2: smartcut / re-encode   (smart_cut.py)
     ↓
thumbnails                      (thumbnails.py)
     ↓
similarity check                (similarity.py)
     ↓
JSON output
```

---

## Quick Run

```bash
pip install amverge[ml]
python examples/custom-pipeline/full_pipeline.py episode.mp4
```
