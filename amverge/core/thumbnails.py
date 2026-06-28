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


def make_thumbnail(clip_path: str, thumb_path: str) -> bool:
    try:
        with av.open(clip_path) as container:
            if not container.streams.video:
                return False

            stream = container.streams.video[0]
            stream.codec_context.skip_frame = "NONKEY"

            for frame in container.decode(stream):
                image = frame.to_image()
                new_h = max(1, int(THUMB_WIDTH * image.height / image.width))
                image = image.resize((THUMB_WIDTH, new_h), resample=Image.Resampling.LANCZOS)
                image.save(
                    thumb_path, "JPEG",
                    quality=THUMB_QUALITY, optimize=True, progressive=True, subsampling=0,
                )
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
    """Generate thumbnails for all scenes. progress_cb(done, total)."""
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
