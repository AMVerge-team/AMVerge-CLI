"""Stream copy export - lossless, no re-encode.

Reads scenes.json from a detect run and copies selected clips.

Usage:
    python 01_copy_export.py [video_path] [scenes_json]
"""

import sys
import json
import subprocess
from pathlib import Path
from amverge import get_ffmpeg

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "episode.mp4"
SCENES_JSON = sys.argv[2] if len(sys.argv) > 2 else "episode_scenes/scenes.json"

data = json.loads(Path(SCENES_JSON).read_text())
scenes = data.get("scenes", data)

# Fix key name: scenes.json may use "index" or "scene_index"
for s in scenes:
    if "scene_index" not in s and "index" in s:
        s["scene_index"] = s["index"]

# Export first 3 scenes
selected = scenes[:3]
out_dir = Path("export_copy")
out_dir.mkdir(parents=True, exist_ok=True)
ff = get_ffmpeg()

for s in selected:
    idx = s["scene_index"]
    dst = str(out_dir / f"scene_{idx:04d}.mp4")
    subprocess.run(
        [ff, "-y", "-i", s["path"], "-c", "copy", dst],
        capture_output=True, check=True,
    )
    print(f"  Exported scene_{idx:04d}.mp4  ({s.get('duration', s.get('duration_sec', 0)):.2f}s)")

print(f"\n{len(selected)} clips -> {out_dir.resolve()}")
