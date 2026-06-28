"""High-level scene detection API.

Usage::

    from amverge import detect_scenes

    result = detect_scenes("episode.mp4", output_dir="./scenes")

    for scene in result.scenes:
        print(scene.index, scene.start, scene.end, scene.path)

    for a, b in result.similar_pairs:
        print(f"Scenes {a} and {b} look similar")
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Literal

from .core.detection.keyframe import detect_cuts_by_keyframe
from .core.detection.edge import detect_cuts_by_edge
from .core.segmenter import collect_scenes, run_ffmpeg_segment
from .core.thumbnails import generate_thumbnails
from .core.similarity import find_similar_pairs
from .core.video import get_video_duration

DetectionMethod = Literal["keyframe", "edge"]
ProgressCb = Callable[[str, int, str], None]


@dataclass
class Scene:
    index: int
    start: float
    end: float
    duration: float
    path: str
    thumbnail: str | None
    original_file: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DetectResult:
    scenes: list[Scene]
    similar_pairs: list[tuple[int, int]]
    output_dir: str
    scenes_json: str

    def to_dict(self) -> dict:
        return {
            "scenes": [s.to_dict() for s in self.scenes],
            "similar_pairs": [list(p) for p in self.similar_pairs],
            "output_dir": self.output_dir,
            "scenes_json": self.scenes_json,
        }


def detect_scenes(
    video_path: str,
    output_dir: str | None = None,
    method: DetectionMethod = "keyframe",
    min_duration: float = 0.25,
    thumbnails: bool = True,
    similarity: bool = True,
    similarity_threshold: float = 0.10,
    thumbnail_workers: int = 4,
    edge_threshold: float = 0.15,
    edge_radius: float = 0.6,
    edge_blocksize: int = 3,
    progress: ProgressCb | None = None,
) -> DetectResult:
    """Detect scenes in a video file.

    Args:
        video_path: Path to the source video.
        output_dir: Where to write clip files and thumbnails.
            Defaults to ``<video_stem>_scenes/`` next to the video.
        method: Detection method. ``"keyframe"`` cuts at I-frame boundaries
            (fast, lossless). ``"edge"`` uses Canny edges + cosine similarity
            inside keyframe windows (more accurate, slower, requires OpenCV).
        min_duration: Merge any resulting scenes shorter than this many seconds.
        thumbnails: Generate JPEG thumbnails for each scene.
        similarity: Run adjacent-scene similarity check (requires thumbnails).
        similarity_threshold: Cosine dissimilarity below which two adjacent
            scenes are flagged as similar (lower = stricter).
        thumbnail_workers: Number of parallel thumbnail worker threads.
        edge_threshold: Dissimilarity threshold for edge detection cuts.
        edge_radius: Seconds around each keyframe to scan (edge method only).
        edge_blocksize: Pooling block size for edge maps (edge method only).
        progress: Optional callback ``(stage, percent, message)`` receiving
            pipeline stage name, 0-100 percent, and a human-readable message.

    Returns:
        :class:`DetectResult` with scenes, similar pairs, output directory,
        and the path to the saved ``scenes.json`` file.
    """
    video_path = str(Path(video_path).resolve())
    video_stem = Path(video_path).stem

    if output_dir is None:
        output_dir = str(Path(video_path).parent / f"{video_stem}_scenes")

    os.makedirs(output_dir, exist_ok=True)

    def _progress(stage: str, pct: int, msg: str) -> None:
        if progress:
            try:
                progress(stage, pct, msg)
            except Exception:
                pass

    # --- Stage: detect cuts ---
    _progress("detect", 0, f"Starting {method} detection...")

    def _kf_cb(pct: int, msg: str) -> None:
        _progress("detect", pct, msg)

    if method == "keyframe":
        cut_points = detect_cuts_by_keyframe(
            video_path,
            min_duration=min_duration,
            progress_cb=_kf_cb,
        )
    else:
        cut_points = detect_cuts_by_edge(
            video_path,
            threshold=edge_threshold,
            radius=edge_radius,
            blocksize=edge_blocksize,
            min_duration=min_duration,
            progress_cb=_kf_cb,
        )

    _progress("detect", 100, f"Detection done - {len(cut_points)} cuts")

    # --- Stage: segment ---
    _progress("segment", 0, f"Cutting {len(cut_points)} scenes...")

    seg_stem = video_stem.replace("%", "%%")
    output_pattern = os.path.join(output_dir, f"{seg_stem}_%04d.mp4")

    run_ffmpeg_segment(video_path, output_pattern, cut_points)

    total_duration = get_video_duration(video_path)
    raw_scenes = collect_scenes(output_dir, video_stem, cut_points, total_duration)

    _progress("segment", 100, f"{len(raw_scenes)} scenes written")

    scenes = [
        Scene(
            index=s["scene_index"],
            start=s["start"],
            end=s["end"],
            duration=s["duration"],
            path=s["path"],
            thumbnail=s.get("thumbnail"),
            original_file=s["original_file"],
        )
        for s in raw_scenes
    ]

    similar_pairs: list[tuple[int, int]] = []

    # --- Stage: thumbnails ---
    if thumbnails and scenes:
        _progress("thumbnails", 0, f"Generating {len(scenes)} thumbnails...")
        total = len(scenes)

        def _thumb_cb(done: int, t: int) -> None:
            _progress("thumbnails", int(100 * done / max(t, 1)), f"Thumbnails {done}/{t}")

        generate_thumbnails(
            raw_scenes,
            output_dir,
            video_stem,
            workers=thumbnail_workers,
            progress_cb=_thumb_cb,
        )

        for scene in scenes:
            scene.thumbnail = os.path.join(output_dir, f"{video_stem}_{scene.index:04d}.jpg")

        _progress("thumbnails", 100, "Thumbnails done")

        # --- Stage: similarity ---
        if similarity:
            _progress("similarity", 0, "Checking adjacent scene similarity...")
            total_pairs = max(len(scenes) - 1, 1)

            def _sim_cb(done: int, t: int) -> None:
                _progress("similarity", int(100 * done / max(t, 1)), f"Pairs {done}/{t}")

            similar_pairs = find_similar_pairs(
                [s.to_dict() for s in scenes],
                threshold=similarity_threshold,
                progress_cb=_sim_cb,
            )

            _progress("similarity", 100, f"Found {len(similar_pairs)} similar pairs")

    # --- Save JSON ---
    scenes_json_path = os.path.join(output_dir, f"{video_stem}_scenes.json")
    payload = {
        "scenes": [s.to_dict() for s in scenes],
        "similar_pairs": [list(p) for p in similar_pairs],
    }
    Path(scenes_json_path).write_text(json.dumps(payload, indent=2))

    return DetectResult(
        scenes=scenes,
        similar_pairs=similar_pairs,
        output_dir=output_dir,
        scenes_json=scenes_json_path,
    )
