from __future__ import annotations

import json
import shutil
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import typer

from ...core.infra.ipc import emit_event, emit_progress, log

_CUT_MODES = ("reencode", "auto", "copy")


def _plan_copy_items(items: list[dict], cut_mode: str) -> dict[int, list[float]]:
    """Decide which range items stream-copy, mirroring ``backend``'s choice.

    ``copy`` copies every range (keyframe-detection episodes, whose scenes all
    start on keyframes). ``auto`` copies only what
    :func:`split_exact_copy_scenes` proves bleed-free, on sources
    :func:`exact_copy_supported` accepts (AI-detected episodes). Keyframes are
    read once per source, not per item.

    Returns ``{item_index: keyframes}`` for the items to copy.
    """
    if cut_mode == "reencode":
        return {}

    from ...core.cutting.smart_cut import exact_copy_supported, split_exact_copy_scenes
    from ...core.keyframes.keyframe_align import get_keyframe_timestamps_pyav, get_open_gop_keyframes
    from ...core.video.probe_utils import probe_video_duration, probe_video_rate

    by_source: dict[str, list[int]] = {}
    for i, item in enumerate(items):
        if not item.get("existing_clip_path") and item.get("source_path"):
            by_source.setdefault(item["source_path"], []).append(i)

    plan: dict[int, list[float]] = {}
    for source, indexes in by_source.items():
        src = Path(source)
        try:
            if cut_mode == "auto" and not exact_copy_supported(src):
                continue
            keyframes = get_keyframe_timestamps_pyav(source)
            if not keyframes:
                continue
            if cut_mode == "copy":
                for i in indexes:
                    plan[i] = keyframes
                continue
            scenes = sorted(
                (
                    {"_i": i, "start_sec": float(items[i]["start_sec"]), "end_sec": float(items[i]["end_sec"])}
                    for i in indexes
                ),
                key=lambda s: s["start_sec"],
            )
            copy, _ = split_exact_copy_scenes(
                scenes,
                keyframes,
                get_open_gop_keyframes(source),
                probe_video_rate(src),
                probe_video_duration(src),
            )
            for scene in copy:
                plan[scene["_i"]] = keyframes
        except Exception as error:  # noqa: BLE001 — planning failure only costs speed, re-encode stays correct
            log(f"materialize_clips: copy check failed for {src.name}, re-encoding: {error}")
    return plan


def _materialize_one(
    batch_index: int,
    item: dict,
    out_dir: Path,
    copy_keyframes: list[float] | None,
) -> dict:
    from ...core.thumbnails import make_thumbnail
    from ...core.cutting.smart_cut import cut_scene

    stem = uuid.uuid4().hex
    dest_thumb = out_dir / f"{stem}.jpg"
    # cut_scene names its output scene_<index>, so each cut gets a private folder:
    # out_dir may already hold real scene_XXXX clips (an episode's scenes/ on resume)
    work_dir = out_dir / f".materialize_{stem}"

    try:
        existing_clip = item.get("existing_clip_path")
        if existing_clip:
            dest_clip = out_dir / f"{stem}.mp4"
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
                "clip_mode": "copy",
                "error": None,
            }

        source_path = item["source_path"]
        start_sec = float(item["start_sec"])
        end_sec = float(item["end_sec"])

        work_dir.mkdir(parents=True, exist_ok=True)
        cut_path, mode, poster_offset_sec = cut_scene(
            Path(source_path),
            start_sec,
            end_sec,
            batch_index,
            work_dir,
            copy_keyframes or [],
            "copy" if copy_keyframes else "reencode",
            use_cuda=False,
        )
        # a copy keeps the container its streams fit (.mov/.mkv/.avi), not always .mp4
        dest_clip = out_dir / f"{stem}{Path(cut_path).suffix}"
        Path(cut_path).replace(dest_clip)

        thumb_ok = make_thumbnail(str(dest_clip), str(dest_thumb), seek_sec=poster_offset_sec)

        return {
            "index": batch_index,
            "clip_path": str(dest_clip),
            "thumbnail_path": str(dest_thumb) if thumb_ok else None,
            "clip_mode": mode,
            "error": None,
        }
    except Exception as error:  # noqa: BLE001 — report per-item failure, don't abort the batch
        log(f"materialize_clips: item {batch_index} failed: {error}")
        return {"index": batch_index, "clip_path": None, "thumbnail_path": None, "clip_mode": None, "error": str(error)}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def materialize_clips(
    inputs_json: Path = typer.Option(..., "--inputs-json", help="JSON array of items to materialize"),
    output_dir: str = typer.Option(..., "--output-dir", help="Destination directory"),
    cut_mode: str = typer.Option(
        "reencode",
        "--cut-mode",
        help="Range items: 'reencode' (always), 'auto' (copy when keyframe-exact), 'copy' (always copy)",
    ),
) -> None:
    """
    Cut or copy a batch of scenes into a destination folder, each producing a
    standalone clip + poster .jpg. Used for Scenepack storage and for resuming
    an episode import that stopped early.

    Called by Rust exactly like:
        amverge materialize-clips --inputs-json <tmpfile> --output-dir <dir> [--cut-mode auto]

    Each input item is either an already-cut clip file to copy in (video-mode
    source: ``{"existing_clip_path", "existing_thumbnail_path"}``), or a
    ``[start_sec, end_sec]`` range to cut from a source episode (webp-mode
    source: ``{"source_path", "start_sec", "end_sec"}``). Ranges re-encode by
    default so their preview begins at the requested scene boundary;
    ``--cut-mode auto|copy`` stream-copies them the way ``backend`` would.

    Emits IPC progress plus one ``MATERIALIZED|<json>`` event per finished
    item to stderr, and the final JSON to stdout.
    """
    if cut_mode not in _CUT_MODES:
        raise typer.BadParameter(f"--cut-mode must be one of {', '.join(_CUT_MODES)}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = json.loads(inputs_json.read_text(encoding="utf-8"))
    if not items:
        print(json.dumps({"schema_version": "1.0", "items": [], "error": None}), flush=True)
        return

    total = len(items)
    emit_progress(0, f"Adding {total} clip(s) to Scenepack...")
    copy_plan = _plan_copy_items(items, cut_mode)
    if cut_mode != "reencode":
        log(f"materialize_clips: {len(copy_plan)} copy, {total - len(copy_plan)} other ({cut_mode})")

    results: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_materialize_one, i, item, out_dir, copy_plan.get(i)): i
            for i, item in enumerate(items)
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            emit_event(f"MATERIALIZED|{json.dumps(result)}")
            done += 1
            emit_progress(int(done / total * 100), f"Added {done}/{total} clip(s)")

    print(json.dumps({"schema_version": "1.0", "items": results, "error": None}), flush=True)
