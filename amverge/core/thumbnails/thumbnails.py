"""Thumbnail generation from clip files."""
from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

import av
from PIL import Image

THUMB_WIDTH = 960
THUMB_QUALITY = 95


def make_thumbnail(clip_path: str, thumb_path: str, seek_sec: float = 0.0) -> bool:
    """Generate a JPEG thumbnail from a video clip.

    Decodes the frame at ``seek_sec`` into the clip, resizes to
    ``THUMB_WIDTH`` (960px) preserving aspect ratio, and saves as a
    progressive JPEG.

    ``seek_sec`` matters because a copy-mode clip's frame 0 is not
    necessarily the scene's true first frame: ``cutting.smart_cut`` widens
    copy-mode cuts outward to the nearest enclosing keyframes, so the clip
    can open with a stretch of bleed from the *previous* scene (see its
    module docs). Callers pass ``cut_scene``'s own ``poster_offset_sec`` for
    that clip so the poster shows the right content; 0.0 (the default) is
    correct for re-encoded clips, which never have that bleed. When
    ``seek_sec`` is 0, this is a plain decode from the start; otherwise it
    seeks to the nearest keyframe at or before the target and decodes
    forward to it, so the result is exact regardless of GOP structure.

    Args:
        clip_path: Path to the source video clip (any FFmpeg-supported format).
        thumb_path: Output path for the thumbnail JPEG.
        seek_sec: How far into the clip the representative frame is.

    Returns:
        True if a thumbnail was written, False otherwise (no video stream,
        decode error, etc.).

    Example:
        >>> make_thumbnail("scene_0001.mp4", "scene_0001.jpg")
        True
    """
    def _save(image) -> None:
        new_h = max(1, int(THUMB_WIDTH * image.height / image.width))
        image = image.resize((THUMB_WIDTH, new_h), resample=Image.Resampling.LANCZOS)
        image.save(
            thumb_path, "JPEG",
            quality=THUMB_QUALITY, optimize=True, progressive=True, subsampling=0,
        )

    try:
        with av.open(clip_path) as container:
            if not container.streams.video:
                return False
            stream = container.streams.video[0]
            if seek_sec > 0.0:
                offset = int(seek_sec / stream.time_base)
                container.seek(offset, stream=stream)
            for frame in container.decode(stream):
                if seek_sec > 0.0 and frame.time is not None and frame.time < seek_sec:
                    continue
                _save(frame.to_image())
                return True
        return False
    except Exception:
        return False


def generate_thumbnails(
    scenes: list[dict[str, Any]],
    output_dir: str,
    file_name: str,
    workers: int = 4,
    progress_cb: Callable[[int, int], None] | None = None,
) -> None:
    """Generate thumbnails for all scenes using a thread pool.

    Scenes dicts must have a ``"scene_index"`` key. Thumbnail files are
    named ``{file_name}_{index:04d}.jpg`` in ``output_dir``. Clips are
    expected at ``{output_dir}/{file_name}_{index:04d}.mp4``.

    Args:
        scenes: List of scene dicts with ``"scene_index"`` key.
        output_dir: Directory containing clip files and output thumbnails.
        file_name: Base name for thumbnails (usually the video stem).
        workers: Max thread count, capped at ``os.cpu_count() or 4``.
        progress_cb: Optional ``callback(done: int, total: int)`` called
            after each thumbnail completes.

    Example:
        >>> scenes = [{"scene_index": 0}, {"scene_index": 1}]
        >>> generate_thumbnails(scenes, "./out", "episode", workers=4)
    """
    total = len(scenes)
    if total == 0:
        return

    done_count = 0
    lock = threading.Lock()

    def build_one(scene: dict) -> None:
        nonlocal done_count
        idx = scene["scene_index"]
        clip_path = os.path.join(output_dir, f"{file_name}_{idx:04d}.mp4")
        thumb_path = os.path.join(output_dir, f"{file_name}_{idx:04d}.jpg")

        if os.path.exists(clip_path):
            make_thumbnail(clip_path, thumb_path)

        with lock:
            done_count += 1
            if progress_cb:
                try:
                    progress_cb(done_count, total)
                except Exception:
                    pass

    max_workers = min(workers, os.cpu_count() or 4)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(build_one, s) for s in scenes]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception:
                pass
