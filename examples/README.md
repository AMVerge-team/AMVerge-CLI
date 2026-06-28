# Examples

Runnable Python scripts showing how to use AMVerge as a library.
Each subdirectory covers one feature area with its own README.

---

## How to Run

Install AMVerge first:

```bash
pip install amverge[ml]        # TransNetV2 examples need [ml]
pip install amverge[ml,edge]   # all examples
```

Then run any script:

```bash
python examples/detect/01_basic_detect.py
```

Scripts expect a video file. Edit the `VIDEO` variable at the top of each file
or pass the video path as a command-line argument.

---

## Directory Map

| directory | what it covers | needs |
|---|---|---|
| [detect/](detect/) | scene detection: keyframe, edge, TransNetV2 | [edge], [ml] for some |
| [export/](export/) | export clips, re-encode, merge, codec selection | - |
| [info-probe/](info-probe/) | video metadata, stream info, probe diagnostics | - |
| [keyframes/](keyframes/) | extract keyframe timestamps, align scenes | - |
| [cutting/](cutting/) | smart cut, ffmpeg segment, lossless copy | - |
| [thumbnails/](thumbnails/) | generate JPEG thumbnails from clips | - |
| [similarity/](similarity/) | detect visually similar adjacent scenes | - |
| [diagnostics/](diagnostics/) | GPU check, version info, health check | - |
| [custom-pipeline/](custom-pipeline/) | full end-to-end pipeline from scratch | [ml] |
