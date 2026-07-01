<p align="center">
  <img src="../assets/AMVerge-CLI.gif" alt="AMVerge CLI" width="1440"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11+-blue?style=flat-square" alt="Python"/>
  <img src="https://img.shields.io/badge/pypi-amverge-22c55e?style=flat-square" alt="PyPI"/>
  <img src="https://img.shields.io/badge/license-GPL--3.0-22c55e?style=flat-square" alt="License"/>
</p>

# Examples

**Runnable Python scripts showing how to use AMVerge as a library.**  
Each subdirectory covers one feature area with its own README, diagrams, and examples.

---

## How to Run

Install AMVerge with the dependencies you need:

```bash
pip install amverge               # basic (keyframe, export, merge, info)
pip install amverge[ml]           # + TransNetV2 ML detection
pip install amverge[interpolation] # + RIFE AI frame interpolation
pip install amverge[flowframes]    # no extra deps (external Flowframes.exe)
pip install amverge[edge]         # + Canny edge detection
pip install amverge[discord]      # + Discord Rich Presence
pip install amverge[ml,edge,discord,upscale,interpolation]  # everything
```

Then run any script:

```bash
python examples/detect/01_basic_detect.py
```

Scripts accept a video path as command-line argument. If none given, defaults to `episode.mp4` in the working directory.

---

## Directory Map

| Directory | What It Covers | Needs |
|---|---|---|
| [detect/](detect/) | scene detection: keyframe, edge, TransNetV2 | [edge], [ml] for some |
| [export/](export/) | export clips, re-encode, merge, codec selection | - |
| [info-probe/](info-probe/) | video metadata, stream info, probe diagnostics | - |
| [keyframes/](keyframes/) | extract keyframe timestamps, align scenes | - |
| [cutting/](cutting/) | smart cut, ffmpeg segment, lossless copy | - |
| [thumbnails/](thumbnails/) | generate JPEG thumbnails from clips | - |
| [similarity/](similarity/) | detect visually similar adjacent scenes | - |
| [diagnostics/](diagnostics/) | GPU check, version info, health check | - |
| [discord-rpc/](discord-rpc/) | Discord Rich Presence status updates | [discord] |
| [upscale/](upscale/) | AI video upscaling: ML, Anime4K shaders, ArtCNN | [upscale] for ML/ArtCNN |
| [interpolation/](interpolation/) | RIFE PyTorch + Flowframes 1.42.0 frame interpolation (free 1.36.0 planned) | [interpolation] for RIFE, Flowframes for FF |
| [custom-pipeline/](custom-pipeline/) | full end-to-end pipeline from scratch | [ml] |

---

## Structure

```txt
examples/
├── detect/
│   ├── README.md
│   ├── 01_basic_detect.py
│   ├── 02_transnetv2_detect.py
│   ├── 03_edge_detect.py
│   └── 04_custom_settings.py
├── export/
│   ├── README.md
│   ├── 01_copy_export.py
│   ├── 02_reencode_export.py
│   └── 03_merge_export.py
├── info-probe/
│   ├── README.md
│   ├── 01_video_info.py
│   ├── 02_probe.py
│   └── 03_hevc_check.py
├── keyframes/
│   ├── README.md
│   ├── 01_extract_keyframes.py
│   └── 02_align_scenes.py
├── cutting/
│   ├── README.md
│   ├── 01_smart_cut.py
│   ├── 02_ffmpeg_segment.py
│   └── 03_single_scene.py
├── thumbnails/
│   ├── README.md
│   └── 01_make_thumbnails.py
├── similarity/
│   ├── README.md
│   └── 01_find_similar.py
├── diagnostics/
│   ├── README.md
│   ├── 01_gpu_check.py
│   └── 02_version_info.py
├── discord-rpc/
│   ├── README.md
│   └── 01_basic_rpc.py
├── upscale/
│   ├── README.md
│   ├── 01_ml_upscale.py
│   ├── 02_anime4k_upscale.py
│   └── 03_manage_models.py
├── interpolation/
│   ├── README.md
│   ├── 01_flowframes_interpolate.py
│   └── 02_rife_interpolate.py
├── custom-pipeline/
│   ├── README.md
│   └── full_pipeline.py
└── README.md
```

---

## See Also

| | |
|---|---|
| [Library API](../docs/library.md) | complete Python library reference |
| [CLI Reference](../docs/cli-reference.md) | all commands and flags |
| [Detection Methods](../docs/detection-methods.md) | method comparison and tuning |
