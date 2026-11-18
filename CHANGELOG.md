# Changelog

## v1.3.4

- **Interpolation architecture & verification**: Swapped backward warping to canonical `hzwer/Practical-RIFE` implementation and added SHA-256 verification hashes for all 19 interpolation models.
- **Model registries cleanup**: Removed unpublished adore.pth model from upscale registry and updated default upscale model to shufflecugan.
- **Storage & attribution updates**: Updated attributions and streamlined model registries.
- **Desktop storage discovery**: Standalone CLI launches now follow the desktop app's `paths.json` and legacy `ai_storage_path.txt` for moved config, model weights, FFmpeg, and Hugging Face cache. Explicit `AMVERGE_CONFIG_DIR`, `HF_HOME`, and `HF_HUB_CACHE` still take precedence.

## v1.3.3

- **Scene Scout thread-safety fix**: Moved lazy `torch` and `transformers` (`AutoModel`, `AutoProcessor`) imports inside `_LOAD_LOCK` in `amverge/core/scenescout/embedding.py` so the daemon's `_prewarm` thread and `_BackgroundEmbedder` cannot race on `transformers`' lazy importer (`ImportError: cannot import name 'AutoModel' from 'transformers'`).
- **Configurable CLI config/models directory**: Added `AMVERGE_CONFIG_DIR` support to `get_amverge_config_dir()` in `amverge/core/infra/config.py` (with unit tests in `tests/test_config_paths.py`) so desktop installs can place depth/upscale weights outside `%APPDATA%`.

## v1.3.2

- **UTF-8 subprocess pipe decoding**: Added `encoding="utf-8", errors="replace"` across FFmpeg/FFprobe/nvidia-smi subprocess pipes (`engine.py`, `smart_cut.py`, `segmenter.py`, `proxy.py`, `probe_utils.py`) so non-ASCII filenames or source metadata no longer crash Windows `cp1252` readers during WebP/source exports.

## v1.3.1

- **Exact-copy & Scene Scout performance**: Exact-copy merge improvements, windowed keyframe scan, batched Scene Scout links, and two-phase keyframe stream-copy split detection.
