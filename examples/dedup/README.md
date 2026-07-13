# Dedup Examples

Remove duplicate / dead frames from video.

Requirements:

- `pip install amverge[dedup]` for OpenCV-based methods (SSIM, FrameDiff, Advanced)
- FFmpeg method works without any extra deps

## Examples

| File | What It Does |
|------|-------------|
| `01_basic_dedup.py` | Auto-detect method + presets (aggressive/anime/normal/gentle) |
| `02_advanced_dedup.py` | Direct method control, dry-run + CSV frame export |

### Basic Dedup with Presets

```bash
python 01_basic_dedup.py my_video.mp4          # normal preset
python 01_basic_dedup.py my_video.mp4 anime    # anime-optimized
python 01_basic_dedup.py my_video.mp4 aggressive
python 01_basic_dedup.py my_video.mp4 gentle
```

Auto-detects: Advanced (OpenCV installed) → SSIM → FFmpeg fallback.

### Advanced with Dry-Run

```bash
python 02_advanced_dedup.py my_video.mp4 advanced
```

Analyzes frame-by-frame, exports CSV of kept/removed ranges, no encoding.
Review the CSV, then re-run with `dry_run=False` to encode.

## CLI Equivalents

```bash
amverge dedup video.mp4                          # auto method, normal
amverge dedup video.mp4 --anime                  # anime preset
amverge dedup video.mp4 --aggressive --dry-run   # preview aggressive
amverge dedup video.mp4 --gentle -c h265_main10  # output codec
amverge dedup video.mp4 --export-frames out.csv  # frame CSV
```

## Presets

| Preset | Best For |
|--------|----------|
| anime | Animation (on-twos/threes cadence, flat colors) |
| aggressive | Clean animation, maximum dedup |
| normal | Balanced, safe defaults |
| gentle | Grainy footage, live-action, safest |

## Methods

| Method | Deps | Best For |
|--------|------|----------|
| Advanced | OpenCV | Best accuracy, anime cadence detection |
| SSIM | OpenCV | Perceptually-aware, quality-focused |
| FrameDiff | OpenCV | Fast pixel-motion, adaptive threshold |
| FFmpeg | none | Quick, no deps, good for VFR sources |
