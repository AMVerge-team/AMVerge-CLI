from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from .thumbnails import make_thumbnail
from ..similarity.similarity import check_pair_similar
from ..infra.ipc import emit_event, emit_progress

INITIAL_THUMB_THRESHOLD = 24


def generate_thumbnails_streaming(
    output_dir: str,
    scenes: list[dict[str, Any]],
    file_name: str,
) -> None:
    total = len(scenes)

    if total == 0:
        emit_event("INITIAL_CLIPS_READY", json.dumps([]))
        emit_event("PROCESSING_COMPLETE")
        return

    threshold = min(INITIAL_THUMB_THRESHOLD, total)
    position_ready = [False] * total
    initial_emitted = [False]
    next_pair_pos = [0]
    lock = threading.Lock()

    def thumb_path_for(scene: dict) -> str:
        return os.path.join(output_dir, f"{file_name}_{scene['scene_index']:04d}.jpg")

    def try_advance_pairs_locked() -> None:
        if not initial_emitted[0]:
            return
        while next_pair_pos[0] < total - 1:
            pa = next_pair_pos[0]
            pb = pa + 1
            if not (position_ready[pa] and position_ready[pb]):
                break
            sa = scenes[pa]
            sb = scenes[pb]
            similar = check_pair_similar(thumb_path_for(sa), thumb_path_for(sb))
            emit_event("PAIR_RESULT", f"{pa}|{pb}|{'1' if similar else '0'}")
            next_pair_pos[0] += 1

    def try_emit_initial_locked() -> None:
        if initial_emitted[0]:
            return
        if not all(position_ready[:threshold]):
            return
        scenes_json = [
            {**s, "thumbnail_ready": position_ready[i]}
            for i, s in enumerate(scenes)
        ]
        emit_event("INITIAL_CLIPS_READY", json.dumps(scenes_json))
        initial_emitted[0] = True

    def build_one(args: tuple[int, dict]) -> None:
        pos, scene = args
        scene_index = scene["scene_index"]
        clip_path = os.path.join(output_dir, f"{file_name}_{scene_index:04d}.mp4")
        t_path = thumb_path_for(scene)

        if os.path.exists(clip_path):
            make_thumbnail(clip_path, t_path)
            emit_event("THUMBNAIL_READY", str(pos))

        with lock:
            position_ready[pos] = True
            try_emit_initial_locked()
            try_advance_pairs_locked()

    progress_step = max(1, total // 25)
    emit_progress(90, f"Generating thumbnails... 0/{total}")
    max_workers = min(4, os.cpu_count() or 4)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(build_one, (i, s)): i for i, s in enumerate(scenes)}
        done_count = 0

        for future in as_completed(futures):
            done_count += 1
            try:
                future.result()
            except Exception:
                pass

            if done_count % progress_step == 0 or done_count == total:
                emit_progress(90, f"Generating thumbnails... {done_count}/{total}")

    with lock:
        if not initial_emitted[0]:
            scenes_json = [
                {**s, "thumbnail_ready": position_ready[i]}
                for i, s in enumerate(scenes)
            ]
            emit_event("INITIAL_CLIPS_READY|" + json.dumps(scenes_json))
            initial_emitted[0] = True

        try_advance_pairs_locked()

    emit_event("PROCESSING_COMPLETE")
