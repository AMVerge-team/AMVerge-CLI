"""Merge selected scenes into a single file.

Reads scenes.json, builds a concat file, and runs ffmpeg concat.

Usage:
    python 03_merge_export.py [video_path] [scenes_json]
"""

import sys
import json
import subprocess
import tempfile
import os
from pathlib import Path
from amverge import get_ffmpeg

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "episode.mp4"
SCENES_JSON = sys.argv[2] if len(sys.argv) > 2 else "episode_scenes/scenes.json"

data = json.loads(Path(SCENES_JSON).read_text())
scenes = data.get("scenes", data)
for s in scenes:
    if "scene_index" not in s and "index" in s:
        s["scene_index"] = s["index"]

selected = scenes[:5]
ff = get_ffmpeg()

with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
    concat_file = f.name
    for s in selected:
        path = s["path"].replace("\\", "/")
        f.write(f"file '{path}'\n")

dst = "export_merged.mp4"
try:
    subprocess.run(
        [ff, "-y", "-f", "concat", "-safe", "0", "-i", concat_file, "-c", "copy", dst],
        capture_output=True, check=True,
    )
    print(f"Merged {len(selected)} scenes -> {dst}")
finally:
    os.unlink(concat_file)
