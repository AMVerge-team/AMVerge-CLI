"""Smart cut - V2 pipeline cutting with automatic mode selection.

Handles copy / snapped_copy / smartcut / reencode automatically.
Runs in parallel with a thread pool.

Usage:
    python 01_smart_cut.py [video_path]
"""

import sys
import torch
from pathlib import Path
from amverge import (
    cut_all_scenes, get_keyframe_timestamps_pyav, check_if_hevc,
)

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "episode.mp4"

kf = get_keyframe_timestamps_pyav(VIDEO)
is_hevc = check_if_hevc(VIDEO)
device = "cuda" if torch.cuda.is_available() else "cpu"

scenes = [
    {"scene_index": 0, "start_sec": 0.0, "end_sec": 5.0},
    {"scene_index": 1, "start_sec": 5.2, "end_sec": 10.0},
]

out_dir = Path("examples_cutting_output") / "scenes"
print(f"Cutting 2 test scenes from {Path(VIDEO).name}")
print(f"  Device: {device}  HEVC: {is_hevc}  GPU: {device == 'cuda'}\n")

results = cut_all_scenes(
    input_file=Path(VIDEO),
    scenes=scenes,
    keyframes=kf,
    out_dir=out_dir,
    use_cuda=(device == "cuda"),
    is_hevc=is_hevc,
    max_workers=2,
    on_ready=lambda r: print(f"  scene {r['scene_index']}: {r['clip_mode']} -> {Path(r['clip_path']).name}"),
)

modes = {r["clip_mode"] for r in results}
print(f"\nDone. Cut modes used: {', '.join(modes)}")
print(f"Output: {out_dir.resolve()}")
