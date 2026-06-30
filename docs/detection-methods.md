# Detection Methods

AMVerge CLI supports three scene detection methods. The right choice depends on your source material.

---

## Keyframe (`--method keyframe`)

**Default. Recommended for most content.**

Extracts I-frame (keyframe) timestamps from the video using PyAV packet demux.
Splits the video at those timestamps with FFmpeg stream copy - no re-encode.

```bash
amverge detect episode.mp4 --method keyframe
```

**How it works:**

```txt
PyAV packet demux
      ↓
I-frame timestamps
      ↓
merge short scenes (< min_duration)
      ↓
ffmpeg -segment_times (stream copy)
```

**Pros:**

- Very fast - no frame decoding on the detection pass
- Lossless output - stream copy only
- Works well for anime and broadcast content (I-frames land on scene cuts)

**Cons:**

- Accuracy depends on encoder. Some encodes place I-frames in unusual positions, creating short or near-duplicate clips
- Use the `--min-duration` flag and the similarity check to clean these up

---

## TransNetV2 (`--method transnetv2`)

**Best accuracy. Requires PyTorch.**

```bash
pip install amverge[ml]
```

```bash
amverge detect episode.mp4 --method transnetv2
```

TransNetV2 is a deep learning model trained specifically for shot boundary detection in video.
It decodes frames at 48x27 resolution and passes them through a convolutional neural network.

**Decode backend (`--decode-method`):**

The transnetv2 method has two decode backends. The flag is ignored by the keyframe and edge methods.

| value | how | platform | notes |
|-------|-----|----------|-------|
| `ffmpeg` (default) | FFmpeg pipe, decode and inference interleaved in one pass | cross-platform | no extra setup |
| `nelux` | Nelux/NVDEC GPU decode into a frame buffer, then inference | Windows | faster; needs Nelux + FFmpeg shared DLLs (`AMVERGE_FFMPEG_BIN`) |

When `--decode-method nelux` is requested, the CLI runs a quick availability smoke test first. If Nelux or its FFmpeg DLLs are missing, it prints a warning and falls back to the `ffmpeg` backend automatically, so the command always completes.

```bash
amverge detect episode.mp4 --method transnetv2 --decode-method nelux
```

**How it works:**

```txt
FFmpeg pipe decode (48x27 rgb24)   [or Nelux/NVDEC GPU decode]
      ↓ (frames ndarray)
TransNetV2 CNN inference (GPU/CPU)
      ↓ (scene boundaries in frames)
convert frames to seconds
      ↓
get keyframe timestamps (PyAV)
      ↓
classify scenes by keyframe alignment
      ↓
Phase 1: lossless copy (keyframe-aligned scenes, max_workers=8)
      ↓
Phase 2: smartcut or re-encode (non-aligned scenes, max_workers=2)
```

**Cut modes:**

| mode | when | method |
|------|------|--------|
| `copy` | scene starts on a keyframe | lossless stream copy |
| `smartcut` | H.264, next keyframe within 90% of scene | encode tiny head + lossless tail, concat |
| `snapped_copy` | HEVC CPU, nearest keyframe within 5s | lossless copy from snapped keyframe |
| `reencode` | fallback | full re-encode with NVENC (GPU) or libx264/libx265 (CPU) |

**Pros:**

- Highly accurate scene boundaries regardless of keyframe placement
- GPU-accelerated (CUDA auto-detected, CPU fallback)
- Scene cache (.npy) - re-opening same video skips re-detection
- Same smart cut pipeline as AMVerge desktop app

**Cons:**

- Slower than keyframe detection (decodes ~300 frames for a typical 24 min episode)
- Requires ~5 GB VRAM for GPU inference (CPU works too, just slower)
- Requires `pip install amverge[ml]`

---

## Edge (`--method edge`)

**More accurate than keyframe. Requires OpenCV.**

```bash
pip install amverge[edge]
```

```bash
amverge detect episode.mp4 --method edge
```

Decodes every frame and runs Canny edge detection + cosine similarity comparison between adjacent frames. Detects cut boundaries by large shifts in edge density.

**How it works:**

```txt
PyAV full decode
      ↓
Canny edge map per frame
      ↓
cosine similarity between adjacent frames
      ↓
flag frames where similarity drops below threshold
      ↓
merge short scenes (< min_duration)
      ↓
ffmpeg -segment_times (stream copy)
```

**Pros:**

- Frame-accurate detection regardless of keyframe placement
- Handles heavily compressed or poorly encoded sources
- No ML dependency

**Cons:**

- Much slower - decodes every frame
- Requires `pip install amverge[edge]` (OpenCV)

---

## Comparison

| | keyframe | transnetv2 | edge |
|---|---|---|---|
| Speed | Fast | Medium | Slow |
| Accuracy | Good | Best | Excellent |
| Lossless output | Yes | Partial (smart cut) | Yes |
| Extra dependency | None | PyTorch | OpenCV |
| GPU support | - | CUDA | - |
| Cache | - | .npy files | - |
| Best for | Anime, broadcast | Any content | Heavily compressed / bad encodes |

---

## Tuning

### `--min-duration N`

Merges any scene shorter than N seconds into the next one.
Default `0.25`. Raise to `0.5` or `1.0` to reduce false cuts.

### `--similarity-threshold N`

Controls what counts as a "similar" pair of adjacent scenes (0.0 - 1.0).
Lower = stricter. Default `0.10`.
Similarity pairs are flagged in the output table but not automatically merged - you decide.

### `--edge-threshold N` (edge method only)

Cosine similarity drop that counts as a cut. Default `0.15`.
Raise to catch subtler cuts; lower to reduce false positives.

### `--edge-radius N` (edge method only)

Window radius around candidate keyframes. Default `0.6`.
