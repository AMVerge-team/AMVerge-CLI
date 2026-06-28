# TODO

Gaps found between AMVerge app settings and CLI coverage (all resolved).

---

## 1. `detect --method transnetv2` ✅

Done. `detect --method transnetv2` wired in `pipeline.py` and `detect.py`.
Uses TransNetV2 decode+inference, keyframe alignment, smart cut (Phase 1 lossless + Phase 2 re-encode).
Guarded with clear ImportError if `[ml]` extra not installed.
See: `pipeline.py:118-212`, `detect.py:25,43,166-256`

---

## 2. `export --audio` missing ✅

Done. `--audio` option added to `export.py`. Values: `copy | aac | aac_320 | pcm16 | pcm24 | flac | alac | opus | mp3 | none`. Each mapped to correct ffmpeg `-c:a` / `-b:a` flags via `AUDIO_FFMPEG` dict.
See: `export.py:19-32,56`

---

## 3. `export --container` missing ✅

Done. `--container` option added to `export.py`. Values: `mp4 | mkv | mov`. Sets output file extension. ProRes codec compatibility enforced (requires mov).
See: `export.py:20,57`

---

## 4. `export --codec` too limited ✅

Done. `--codec` expanded to full AMVerge app codec list (15 codec profiles + aliases). Each mapped to ffmpeg encoder via `CODEC_PROFILES` dict. `--hardware auto | gpu | cpu` option added for NVENC GPU encode (ignored for ProRes). Backward-compat aliases: `h264` → `h264_main`, `hevc`/`h265` → `h265_main`.
See: `export.py:18-38,44-57`
