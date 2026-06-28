# Export

Export selected scenes from a detect run. Copy (lossless) or re-encode
with full codec profile and hardware acceleration support.

---

## How It Works

```txt
scenes.json (from detect)
     ↓
select scenes by index
     ↓
ffmpeg copy (lossless) or re-encode (codec + audio + hardware)
     ↓
.mp4 / .mkv / .mov clips or merged file
```

---

## Codec Profiles

| shortcut | maps to |
|---|---|
| `h264` | `h264_main` |
| `hevc` / `h265` | `h265_main` |

Full list in [docs/cli-reference.md](../../docs/cli-reference.md).

---

## Examples

| file | description |
|---|---|
| [01_copy_export.py](01_copy_export.py) | stream copy export, lossless |
| [02_reencode_export.py](02_reencode_export.py) | re-encode with codec/audio/hardware |
| [03_merge_export.py](03_merge_export.py) | merge selected scenes into one file |

---

## Quick Run

```bash
python examples/export/01_copy_export.py episode.mp4 scenes.json
python examples/export/02_reencode_export.py episode.mp4 scenes.json
python examples/export/03_merge_export.py episode.mp4 scenes.json
```
