"""Smart cut - V2 pipeline cutting, one mode for the whole batch.

`mode="copy"` is a true stream copy, widened outward to the nearest
keyframes. `mode="reencode"` cuts the exact boundary with a real encode.
Runs in parallel with a thread pool.

Usage:
    python 01_smart_cut.py [video_path] [copy|reencode]
"""

import sys
from pathlib import Path
from amverge import cut_all_scenes, get_keyframe_timestamps_pyav, get_gpu_info

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "episode.mp4"
MODE = sys.argv[2] if len(sys.argv) > 2 else "copy"

kf = get_keyframe_timestamps_pyav(VIDEO)
gpu = get_gpu_info()
device = "cuda" if gpu["cuda_available"] else "cpu"

scenes = [
    {"scene_index": 0, "start_sec": 0.0, "end_sec": 5.0},
    {"scene_index": 1, "start_sec": 5.2, "end_sec": 10.0},
]

out_dir = Path("examples_cutting_output") / "scenes"
print(f"Cutting 2 test scenes from {Path(VIDEO).name} (mode={MODE})")
print(f"  Device: {device}  GPU: {device == 'cuda'}\n")

results = cut_all_scenes(
    input_file=Path(VIDEO),
    scenes=scenes,
    keyframes=kf,
    out_dir=out_dir,
    mode=MODE,
    use_cuda=(device == "cuda"),
    max_workers=2,
    on_ready=lambda r: print(f"  scene {r['scene_index']}: {r['clip_mode']} -> {Path(r['clip_path']).name}"),
)

modes = {r["clip_mode"] for r in results}
print(f"\nDone. Cut modes used: {', '.join(modes)}")
print(f"Output: {out_dir.resolve()}")
