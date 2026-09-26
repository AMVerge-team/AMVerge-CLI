"""Full custom pipeline - end-to-end scene detection and cutting.

Replicates `amverge detect --method transnetv2` step by step using
AMVerge classes. Modify any step to build your own custom workflow.

Usage:
    pip install amverge[ml]
    python full_pipeline.py [video_path]
"""

import sys
import json
from pathlib import Path
from amverge import (
    TRANSNET_AVAILABLE,
    AmvergeVideo,
    SceneCache,
    ThumbnailGenerator,
    SimilarityChecker,
    decode_and_detect_scenes,
    scenes_to_objects,
    cut_all_scenes,
    get_gpu_info,
)

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "episode.mp4"

if not TRANSNET_AVAILABLE:
    print("Error: transnetv2_pytorch not installed. Run: pip install amverge[ml]")
    sys.exit(1)

video = AmvergeVideo(VIDEO)
gpu = get_gpu_info()
device = "cuda" if gpu["cuda_available"] else "cpu"

output_dir = video.path.parent / f"{video.stem}_pipeline"
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\n{'='*60}")
print(f"Custom Pipeline: {video.name}")
print(f"  {video.width}x{video.height} {video.fps}fps {video.duration:.1f}s")
print(f"  Device: {device}  Output: {output_dir}")
print(f"{'='*60}\n")

# Step 1: TransNetV2 detection
print("[1/6] Running TransNetV2 scene detection...")
scenes_secs, scenes_frames = decode_and_detect_scenes(video.path)
print(f"       {len(scenes_secs)} scenes detected")
for i, (s, e) in enumerate(scenes_secs[:5]):
    print(f"       scene {i}: {s:.2f}s - {e:.2f}s ({e-s:.2f}s)")
if len(scenes_secs) > 5:
    print(f"       ... and {len(scenes_secs) - 5} more")

cache = SceneCache(output_dir)
cache.save(video.path, scenes_secs, scenes_frames)
print(f"       cached to {output_dir}/\n")

# Step 2: Build scene objects
print("[2/6] Building scene objects...")
raw_scenes = scenes_to_objects(scenes_secs=scenes_secs, scenes_frames=scenes_frames)
print(f"       {len(raw_scenes)} scenes\n")

# Step 3: Cut scenes
print("[3/6] Cutting scenes...")
scenes_out_dir = output_dir / "scenes"
cut_by_idx: dict[int, dict] = {}

def on_clip_ready(result: dict) -> None:
    cut_by_idx[result["scene_index"]] = result
    print(f"       scene {result['scene_index']}: {result['clip_mode']}")

cut_all_scenes(
    input_file=video.path, scenes=raw_scenes, keyframes=video.keyframes,
    out_dir=scenes_out_dir, mode="reencode", use_cuda=(device == "cuda"),
    max_workers=2, on_ready=on_clip_ready,
)
print()

# Step 4: Thumbnails - seek past any keyframe-snap bleed (copy mode only)
print("[4/6] Generating thumbnails...")
gen = ThumbnailGenerator(workers=4)
scene_thumb_dicts = []
for s in raw_scenes:
    idx = s["scene_index"]
    cut = cut_by_idx.get(idx, {})
    clip_path = cut.get("clip_path", "")
    thumb_path = output_dir / f"{video.stem}_{idx:04d}.jpg"
    if clip_path and Path(clip_path).exists():
        gen.generate_one(clip_path, thumb_path, seek_sec=cut.get("poster_offset_sec") or 0.0)
    scene_thumb_dicts.append({
        "scene_index": idx,
        "thumbnail": str(thumb_path) if thumb_path.exists() else None,
    })
print(f"       done\n")

# Step 5: Similarity check
print("[5/6] Checking similarity...")
checker = SimilarityChecker(threshold=0.10)
pairs = checker.find_in([s for s in scene_thumb_dicts if s["thumbnail"]])
print(f"       {len(pairs)} similar pair(s) found")
if pairs:
    for a, b in pairs:
        print(f"       scenes {a} and {b} look similar")
print()

# Step 6: Build manifest
print("[6/6] Building manifest...")
manifest = {
    "video": video.to_dict(),
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

manifest_path = output_dir / f"{video.stem}_pipeline.json"
manifest_path.write_text(json.dumps(manifest, indent=2))
print(f"\n{'='*60}")
print(f"Pipeline complete! {len(raw_scenes)} scenes, {len(pairs)} similar pairs")
print(f"Output: {output_dir.resolve()}")
print(f"Manifest: {manifest_path}")
print(f"{'='*60}")
