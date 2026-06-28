"""Re-encode export with codec profile, audio, and hardware selection.

Shows how to use CODEC_PROFILES and AUDIO_FFMPEG mappings from the library.

Usage:
    python 02_reencode_export.py [video_path] [scenes_json]
"""

import sys
import json
import subprocess
from pathlib import Path
from amverge import (
    get_ffmpeg, check_if_hevc,
    CODEC_PROFILES, CODEC_ALIASES, AUDIO_FFMPEG,
)
from amverge.core.codec_utils import _resolve_gpu

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "episode.mp4"
SCENES_JSON = sys.argv[2] if len(sys.argv) > 2 else "episode_scenes/scenes.json"

data = json.loads(Path(SCENES_JSON).read_text())
scenes = data.get("scenes", data)
for s in scenes:
    if "scene_index" not in s and "index" in s:
        s["scene_index"] = s["index"]

codec_name = "h264_main"
audio_name = "aac_320"
hardware = "auto"
container = "mp4"

codec_name = CODEC_ALIASES.get(codec_name, codec_name)
use_gpu = _resolve_gpu(hardware, codec_name)
profile = CODEC_PROFILES[codec_name]
encoder = profile["gpu"] if use_gpu and profile["gpu"] else profile["cpu"]
audio_args = AUDIO_FFMPEG[audio_name]

desc = f"{codec_name} ({encoder}) {audio_name} hw={hardware} gpu={use_gpu}"
print(f"Re-encoding first scene with {desc}\n")

selected = scenes[:1]
out_dir = Path("export_reencode")
out_dir.mkdir(parents=True, exist_ok=True)
ff = get_ffmpeg()

for s in selected:
    idx = s["scene_index"]
    dst = str(out_dir / f"scene_{idx:04d}.{container}")
    cmd = [ff, "-y", "-i", s["path"], "-c:v", str(encoder)]
    args = str(profile.get("args", "")).strip()
    if args:
        cmd += args.split()
    cmd += audio_args
    cmd.append(dst)
    subprocess.run(cmd, capture_output=True, check=True)
    print(f"  Encoded scene_{idx:04d}.{container}")

print(f"\nDone -> {out_dir.resolve()}")
