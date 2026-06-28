"""Full custom pipeline - end-to-end scene detection and cutting.

Replicates `amverge detect --method transnetv2` step by step using
only the low-level library API. Modify any step to build your own
custom detection or cutting logic.

Usage:
    pip install amverge[ml]
    python full_pipeline.py [video_path]
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from amverge import (
    TRANSNET_AVAILABLE,
    decode_and_detect_scenes,
    get_keyframe_timestamps_pyav,
    classify_scenes_by_keyframe_alignment,
    scenes_to_objects,
    check_if_hevc,
    cut_all_scenes,
    make_thumbnail,
    find_similar_pairs,
    probe_video_duration,
    probe_video_fps,
    probe_video_dimensions,
)

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "episode.mp4"

if not TRANSNET_AVAILABLE:
    print("Error: transnetv2_pytorch not installed. Run: pip install amverge[ml]")
    sys.exit(1)

video_path = Path(VIDEO).resolve()
video_stem = video_path.stem
output_dir = video_path.parent / f"{video_stem}_pipeline"
output_dir.mkdir(parents=True, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n{'='*60}")
print(f"Custom Pipeline: {video_path.name}")
print(f"Device: {device}  Output: {output_dir}")
print(f"{'='*60}\n")

# Step 1: TransNetV2 detection
print("[1/7] Running TransNetV2 scene detection...")
scenes_secs, scenes_frames = decode_and_detect_scenes(video_path)
print(f"       {len(scenes_secs)} scenes detected")
for i, (s, e) in enumerate(scenes_secs[:5]):
    print(f"       scene {i}: {s:.2f}s - {e:.2f}s ({e-s:.2f}s)")
if len(scenes_secs) > 5:
    print(f"       ... and {len(scenes_secs) - 5} more")

# Cache results
cache_prefix = f"pipeline_{video_stem}"
np.save(output_dir / f"{cache_prefix}_secs.npy", scenes_secs)
np.save(output_dir / f"{cache_prefix}_frames.npy", scenes_frames)
print(f"       cached to {output_dir}/{cache_prefix}_*.npy\n")

# Step 2: Keyframe extraction
print("[2/7] Extracting keyframe timestamps...")
keyframes = get_keyframe_timestamps_pyav(str(video_path))
print(f"       {len(keyframes)} keyframes found\n")

# Step 3: HEVC check
print("[3/7] Checking codec...")
is_hevc = check_if_hevc(str(video_path))
print(f"       HEVC: {is_hevc}\n")

# Step 4: Build scene objects
print("[4/7] Building scene objects...")
raw_scenes = scenes_to_objects(scenes_secs=scenes_secs, scenes_frames=scenes_frames)

# Step 5: Classify scenes for copying vs re-encoding
print("[5/7] Classifying scenes by keyframe alignment...")
scene_pairs = [(s["start_sec"], s["end_sec"]) for s in raw_scenes]
copy_candidates, reencode_candidates = classify_scenes_by_keyframe_alignment(
    scene_pairs, keyframes
)
copy_idx = {c["scene_id"] for c in copy_candidates}
phase1 = [s for s in raw_scenes if s["scene_index"] in copy_idx]
phase2 = [s for s in raw_scenes if s["scene_index"] not in copy_idx]
print(f"       Phase 1 (lossless copy): {len(phase1)} scenes")
print(f"       Phase 2 (re-encode):    {len(phase2)} scenes\n")

# Step 6: Cut scenes
print("[6/7] Cutting scenes...")
scenes_out_dir = output_dir / "scenes"
cut_by_idx: dict[int, dict] = {}

def on_clip_ready(result: dict) -> None:
    cut_by_idx[result["scene_index"]] = result
    print(f"       scene {result['scene_index']}: {result['clip_mode']}")

if phase1:
    print(f"       Phase 1: {len(phase1)} lossless copies...")
    cut_all_scenes(
        input_file=video_path, scenes=phase1, keyframes=keyframes,
        out_dir=scenes_out_dir, use_cuda=(device == "cuda"),
        is_hevc=is_hevc, max_workers=8, on_ready=on_clip_ready,
    )

if phase2:
    print(f"       Phase 2: {len(phase2)} re-encodes...")
    cut_all_scenes(
        input_file=video_path, scenes=phase2, keyframes=keyframes,
        out_dir=scenes_out_dir, use_cuda=(device == "cuda"),
        is_hevc=is_hevc, max_workers=2, on_ready=on_clip_ready,
        emit_progress_updates=False,
    )
print()

# Step 7: Thumbnails + similarity
print("[7/7] Generating thumbnails and checking similarity...")
scene_objects = []
for s in raw_scenes:
    idx = s["scene_index"]
    cut = cut_by_idx.get(idx, {})
    clip_path = cut.get("clip_path", "")
    thumb_path = str(output_dir / f"{video_stem}_{idx:04d}.jpg")
    if clip_path and Path(clip_path).exists():
        make_thumbnail(clip_path, thumb_path)
    scene_objects.append({
        "scene_index": idx,
        "thumbnail": thumb_path if Path(thumb_path).exists() else None,
    })

pairs = find_similar_pairs(
    [s for s in scene_objects if s["thumbnail"]],
    threshold=0.10,
)
print(f"       {len(pairs)} similar pair(s) found")
if pairs:
    for a, b in pairs:
        print(f"       scenes {a} and {b} look similar")

# Final: video metadata
dur = probe_video_duration(video_path)
fps = probe_video_fps(video_path)
w, h = probe_video_dimensions(video_path)

manifest = {
    "video": {
        "path": str(video_path),
        "duration": dur,
        "width": w,
        "height": h,
        "fps": fps,
    },
    "scenes": [
        {
            "index": s["scene_index"],
            "start": s["start_sec"],
            "end": s["end_sec"],
            "duration": s["duration_sec"],
            "path": cut_by_idx.get(s["scene_index"], {}).get("clip_path", ""),
            "mode": cut_by_idx.get(s["scene_index"], {}).get("clip_mode", "failed"),
        }
        for s in raw_scenes
    ],
    "similar_pairs": [list(p) for p in pairs],
}

manifest_path = output_dir / f"{video_stem}_pipeline.json"
manifest_path.write_text(json.dumps(manifest, indent=2))
print(f"\n{'='*60}")
print(f"Pipeline complete! {len(raw_scenes)} scenes, {len(pairs)} similar pairs")
print(f"Output: {output_dir.resolve()}")
print(f"Manifest: {manifest_path}")
print(f"{'='*60}")
