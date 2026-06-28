"""Similar scene pair detection.

Compares adjacent thumbnails using cosine similarity on 8x8 pooled pixels.
Flags pairs below the dissimilarity threshold.

Usage:
    python 01_find_similar.py [video_path]
"""

import sys
from pathlib import Path
from amverge import detect_scenes, check_pair_similar, find_similar_pairs

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "episode.mp4"

result = detect_scenes(VIDEO, method="keyframe", thumbnails=True, similarity=False)

# Convert Scene objects to dicts for find_similar_pairs
scene_dicts = [s.to_dict() for s in result.scenes]

print(f"\nChecking {len(scene_dicts)} scenes for similarity...")
pairs = find_similar_pairs(scene_dicts, threshold=0.10)

if pairs:
    print(f"\n{len(pairs)} similar pair(s) found:")
    for a, b in pairs:
        similar = check_pair_similar(
            scene_dicts[a]["thumbnail"],
            scene_dicts[b]["thumbnail"],
        )
        print(f"  scenes {a} and {b}  similar={similar}")
else:
    print("No similar pairs found.")

# Manual check between first two thumbnails
if len(scene_dicts) >= 2:
    a_path = scene_dicts[0]["thumbnail"]
    b_path = scene_dicts[1]["thumbnail"]
    if a_path and b_path:
        sim = check_pair_similar(a_path, b_path)
        print(f"\nManual: scenes 0 vs 1 -> similar={sim}")
