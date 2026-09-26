"""Thumbnail generation from video clips.

Usage:
    >>> from amverge import ThumbnailGenerator
    >>> gen = ThumbnailGenerator(workers=4)
    >>> gen.generate_one("clip.mp4", "thumb.jpg")
    >>> gen.generate(scenes, output_dir="./thumbs")
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


class ThumbnailGenerator:
    """Generate JPEG thumbnails from clip files in parallel.

    Decodes the frame at ``seek_sec`` into each clip (the start, by
    default), resizes to 960px wide, and saves as progressive JPEG at 95%
    quality. Pass ``cut_scene``'s own ``poster_offset_sec`` for a clip cut
    in ``"copy"`` mode (see ``cutting.smart_cut``) - its frame 0 can be
    bleed from the previous scene rather than the true start.

    Args:
        workers: Number of parallel threads (capped at CPU count).

    Example:
        >>> gen = ThumbnailGenerator(workers=4)
        >>> gen.generate_one("scene_0001.mp4", "thumb.jpg")
        >>>
        >>> scenes = [{"scene_index": 0}, {"scene_index": 1}]
        >>> gen.generate(scenes, "./thumbs", file_name="episode")
    """

    def __init__(self, workers: int = 4) -> None:
        self._workers = workers

    def generate_one(self, clip_path: str | Path, thumb_path: str | Path, seek_sec: float = 0.0) -> bool:
        """Generate a single thumbnail from a clip.

        Args:
            clip_path: Path to the source video clip.
            thumb_path: Output path for the JPEG thumbnail.
            seek_sec: How far into the clip the representative frame is.

        Returns:
            True on success, False on failure.
        """
        from ..thumbnails.thumbnails import make_thumbnail
        return make_thumbnail(str(clip_path), str(thumb_path), seek_sec=seek_sec)

    def generate(
        self,
        scenes: list[dict[str, Any]],
        output_dir: str | Path,
        *,
        file_name: str = "scene",
    ) -> list[str]:
        """Generate thumbnails for all scenes in parallel.

        Scenes dicts must have a ``"scene_index"`` key, and may carry a
        ``"poster_offset_sec"`` key (as returned by ``cut_all_scenes``) to
        seek past any keyframe-snap bleed before grabbing the frame.
        Output files are named ``{file_name}_{index:04d}.jpg``.

        Args:
            scenes: List of scene dicts with ``"scene_index"`` key.
            output_dir: Directory for output thumbnails.
            file_name: Base name for output files.

        Returns:
            Sorted list of output file paths.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        total = len(scenes)
        results: list[str] = []

        def _one(scene: dict) -> str | None:
            idx = scene.get("scene_index", scene.get("index", 0))
            clip_path = scene.get("path", os.path.join(str(out_dir), f"{file_name}_{idx:04d}.mp4"))
            thumb_path = out_dir / f"{file_name}_{idx:04d}.jpg"
            if os.path.exists(clip_path):
                self.generate_one(clip_path, thumb_path, seek_sec=scene.get("poster_offset_sec") or 0.0)
            return str(thumb_path) if thumb_path.exists() else None

        max_w = min(self._workers, total, (os.cpu_count() or 4))
        with ThreadPoolExecutor(max_workers=max(max_w, 1)) as executor:
            futures = {executor.submit(_one, s): s for s in scenes}
            for future in as_completed(futures):
                try:
                    r = future.result()
                    if r:
                        results.append(r)
                except Exception:
                    pass

        return sorted(results)
