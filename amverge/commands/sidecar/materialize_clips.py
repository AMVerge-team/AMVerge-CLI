from __future__ import annotations

import json
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import typer

from ...core.infra.ipc import emit_progress, log
from ...core.thumbnails import make_thumbnail
from ...core.cutting.smart_cut import cut_scene


def _materialize_one(
    batch_index: int,
    item: dict,
    out_dir: Path,
) -> dict:
    stem = uuid.uuid4().hex
    dest_clip = out_dir / f"{stem}.mp4"
    dest_thumb = out_dir / f"{stem}.jpg"

    try:
        existing_clip = item.get("existing_clip_path")
        if existing_clip:
            shutil.copy2(existing_clip, dest_clip)

            existing_thumb = item.get("existing_thumbnail_path")
            if existing_thumb and Path(existing_thumb).exists():
                shutil.copy2(existing_thumb, dest_thumb)
                thumb_ok = True
            else:
                thumb_ok = make_thumbnail(str(dest_clip), str(dest_thumb))

            return {
                "index": batch_index,
                "clip_path": str(dest_clip),
                "thumbnail_path": str(dest_thumb) if thumb_ok else None,
                "error": None,
            }

        source_path = item["source_path"]
        start_sec = float(item["start_sec"])
        end_sec = float(item["end_sec"])

        cut_path, _mode, poster_offset_sec = cut_scene(
            Path(source_path),
            start_sec,
            end_sec,
            batch_index,
            out_dir,
            [],
            "reencode",
            use_cuda=False,
        )
        Path(cut_path).replace(dest_clip)

        thumb_ok = make_thumbnail(str(dest_clip), str(dest_thumb), seek_sec=poster_offset_sec)

        return {
            "index": batch_index,
            "clip_path": str(dest_clip),
            "thumbnail_path": str(dest_thumb) if thumb_ok else None,
            "error": None,
        }
    except Exception as error:  # noqa: BLE001 — report per-item failure, don't abort the batch
        log(f"materialize_clips: item {batch_index} failed: {error}")
        return {"index": batch_index, "clip_path": None, "thumbnail_path": None, "error": str(error)}


def materialize_clips(
    inputs_json: Path = typer.Option(..., "--inputs-json", help="JSON array of items to materialize"),
    output_dir: str = typer.Option(..., "--output-dir", help="Destination directory"),
) -> None:
    """
    Cut or copy a batch of scenes into a Scenepack's own storage folder, each
    producing a standalone .mp4 + poster .jpg.

    Called by Rust exactly like:
        amverge materialize-clips --inputs-json <tmpfile> --output-dir <scene_packs/<id>>

    Each input item is either an already-cut clip file to copy in (video-mode
    source: ``{"existing_clip_path", "existing_thumbnail_path"}``), or a
    ``[start_sec, end_sec]`` range to cut from a source episode (webp-mode
    source: ``{"source_path", "start_sec", "end_sec"}``). Source ranges are
    re-encoded so their preview begins at the requested scene boundary.

    Emits IPC progress to stderr and final JSON to stdout.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = json.loads(inputs_json.read_text(encoding="utf-8"))
    if not items:
        print(json.dumps({"schema_version": "1.0", "items": [], "error": None}), flush=True)
        return

    total = len(items)
    emit_progress(0, f"Adding {total} clip(s) to Scenepack...")

    results: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_materialize_one, i, item, out_dir): i
            for i, item in enumerate(items)
        }
        for future in as_completed(futures):
            results.append(future.result())
            done += 1
            emit_progress(int(done / total * 100), f"Added {done}/{total} clip(s)")

    print(json.dumps({"schema_version": "1.0", "items": results, "error": None}), flush=True)
