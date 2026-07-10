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


def make_thumbnail(clip_path: str, thumb_path: str, first_keyframe: bool = True) -> bool:
    """Generate a JPEG thumbnail from a video clip.

    Decodes one frame, resizes to ``THUMB_WIDTH`` (960px) preserving aspect
    ratio, and saves as a progressive JPEG.

    Args:
        clip_path: Path to the source video clip (any FFmpeg-supported format).
        thumb_path: Output path for the thumbnail JPEG.
        first_keyframe: When True (lossless *copy* clips, which can start
            mid-GOP), skip to the first keyframe — fast, and avoids decoding a
            long run of predicted frames. When False (*re-encoded* clips, whose
            frame 0 is already an IDR at the exact scene start), decode the first
            frame so the poster is the true opening frame of the scene.

    Returns:
        True if a thumbnail was written, False otherwise (no video stream,
        decode error, etc.).

    Example:
        >>> make_thumbnail("scene_0001.mp4", "scene_0001.jpg")  # copy clip
        True
        >>> make_thumbnail("scene_0004.mp4", "scene_0004.jpg", first_keyframe=False)
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
        # Primary pass: keyframe-only for copy clips, full decode for re-encodes.
        with av.open(clip_path) as container:
            if not container.streams.video:
                return False

            stream = container.streams.video[0]
            if first_keyframe:
                stream.codec_context.skip_frame = "NONKEY"

            for frame in container.decode(stream):
                _save(frame.to_image())
                return True

        # Fallback: the keyframe-only pass decoded nothing (a copy clip that
        # starts mid-GOP with no early keyframe) — decode the first frame.
        if first_keyframe:
            with av.open(clip_path) as container:
                stream = container.streams.video[0]
                for frame in container.decode(stream):
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
