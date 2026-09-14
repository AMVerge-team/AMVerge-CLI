from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import typer

from ...pipeline import detect_scenes, DetectResult
from ...ui import banner, console, err, make_progress, make_table, ok, warn, fail, dim
from ...core.discord.discord_rpc import RPC_AVAILABLE, DiscordRPC

_STAGE_LABELS = {
    "detect":     "Detecting cuts",
    "segment":    "Cutting scenes",
    "thumbnails": "Thumbnails",
    "similarity": "Similarity check",
}


def _notice(ipc: bool, message: str) -> None:
    """A non-fatal warning that is safe to emit in either mode.

    `warn` prints to stdout, and under ``--ipc`` stdout carries nothing but the
    final JSON document. A warning written there lands in the middle of the
    payload and the caller's parse fails on it, so IPC callers get it on stderr
    with the rest of the event stream instead.
    """
    if ipc:
        from ...core.infra.ipc import log

        log(message)
    else:
        warn(message)


def detect(
    video: Path = typer.Argument(..., help="Input video file", exists=True),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    method: str = typer.Option("keyframe", "--method", "-m", help="keyframe · edge · transnetv2"),
    decode_method: str = typer.Option("ffmpeg", "--decode-method", help="transnetv2 decode: ffmpeg (parallel) · nelux (GPU, Windows)"),
    format: str = typer.Option("table", "--format", "-f", help="table · json · paths"),
    json_output: Optional[Path] = typer.Option(None, "--json-output", help="Save JSON to file"),
    no_thumbnails: bool = typer.Option(False, "--no-thumbnails"),
    no_similarity: bool = typer.Option(False, "--no-similarity"),
    min_duration: float = typer.Option(0.25, "--min-duration"),
    workers: int = typer.Option(4, "--workers"),
    similarity_threshold: float = typer.Option(0.10, "--similarity-threshold"),
    edge_threshold: float = typer.Option(0.15, "--edge-threshold"),
    edge_radius: float = typer.Option(0.6, "--edge-radius"),
    threshold: float = typer.Option(0.5, "--threshold", help="transnetv2 cut confidence 0-1 (lower = more cuts; try 0.3 for dark footage)"),
    ipc: bool = typer.Option(False, "--ipc", hidden=True, help="Emit IPC events for Tauri app"),
    no_rpc: bool = typer.Option(False, "--no-rpc", help="Disable Discord RPC"),
) -> None:
    """Detect scenes in a video file."""
    fmt = format.lower()
    if fmt not in ("table", "json", "paths"):
        fail("--format must be: table, json, or paths")
        raise typer.Exit(1)
    if method not in ("keyframe", "edge", "transnetv2"):
        fail("--method must be: keyframe, edge, or transnetv2")
        raise typer.Exit(1)
    if decode_method not in ("ffmpeg", "nelux"):
        fail("--decode-method must be: ffmpeg or nelux")
        raise typer.Exit(1)
    if not 0.0 < threshold <= 1.0:
        fail("--threshold must be between 0 (exclusive) and 1")
        raise typer.Exit(1)
    if threshold != 0.5 and method != "transnetv2":
        _notice(ipc, "--threshold only applies to --method transnetv2; ignoring")
    if decode_method != "ffmpeg" and method != "transnetv2":
        _notice(ipc, "--decode-method only applies to --method transnetv2; ignoring")
        decode_method = "ffmpeg"
    if method == "transnetv2" and decode_method == "nelux":
        from ...core.detection.nelux_runtime import nelux_available
        if not nelux_available():
            _notice(ipc, "Nelux unavailable, falling back to FFmpeg parallel decode")
            decode_method = "ffmpeg"

    if ipc:
        _detect_ipc(video, output, method, min_duration, workers, similarity_threshold, edge_threshold, edge_radius, threshold, decode_method)
        return

    banner("detect")

    rpc = DiscordRPC() if RPC_AVAILABLE and not no_rpc else None
    if rpc:
        rpc.connect()
        rpc.update_detecting(video.name)

    try:
        with make_progress() as progress:
            tasks: dict[str, object] = {}

            def on_progress(stage: str, pct: int, msg: str) -> None:
                label = _STAGE_LABELS.get(stage, stage)
                if stage not in tasks:
                    tasks[stage] = progress.add_task(label, total=100)
                progress.update(tasks[stage], completed=pct, description=label)
                if rpc and stage == "detect":
                    rpc.update_detecting(video.name, pct)

            result: DetectResult = detect_scenes(
                str(video.resolve()),
                output_dir=str(output.resolve()) if output else None,
                method=method,
                decode_method=decode_method,
                min_duration=min_duration,
                thumbnails=not no_thumbnails,
                similarity=not no_similarity and not no_thumbnails,
                similarity_threshold=similarity_threshold,
                thumbnail_workers=workers,
                edge_threshold=edge_threshold,
                edge_radius=edge_radius,
                ai_threshold=threshold,
                progress=on_progress,
            )

        if rpc:
            rpc.update_complete()
    except Exception:
        if rpc:
            rpc.update_error("Detection failed")
        raise
    finally:
        if rpc:
            rpc.clear_presence()
            rpc.disconnect()

    if not result.scenes:
        fail("No scenes detected.")
        raise typer.Exit(1)

    if json_output:
        json_output.write_text(json.dumps(result.to_dict(), indent=2))
        ok(f"JSON saved to {json_output}")

    similar_set = {idx for pair in result.similar_pairs for idx in pair}

    if fmt == "json":
        console.print_json(json.dumps(result.to_dict()))
        return
    if fmt == "paths":
        for scene in result.scenes:
            console.print(scene.path)
        return

    t = make_table(
        ("#",        "muted",  {"justify": "right", "width": 5}),
        ("Start",    None,     {"justify": "right", "width": 9}),
        ("End",      None,     {"justify": "right", "width": 9}),
        ("Duration", None,     {"justify": "right", "width": 9}),
        ("~",        "warn",   {"justify": "center", "width": 3}),
        title=f"{video.stem}  ·  {len(result.scenes)} scenes  ·  {method}",
    )
    for s in result.scenes:
        t.add_row(
            str(s.index),
            f"{s.start:.2f}s",
            f"{s.end:.2f}s",
            f"{s.duration:.2f}s",
            "~" if s.index in similar_set else "",
        )
    console.print(t)
    dim(f"scenes.json saved to {result.scenes_json}")


def _write_scenes_json(output_dir: str, scenes: list) -> None:
    """Leave the scene list beside the clips so a run can be reopened later.

    The IPC caller gets this same list on stdout, but that is gone once the
    process is. Without a copy on disk, reopening an output directory means
    guessing scene boundaries from filenames and losing every timing.
    """
    try:
        target = os.path.join(output_dir, "scenes.json")
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(scenes, handle)
    except OSError as exc:
        # not fatal: the caller already has the scenes, this only costs reopening
        from ...core.infra.ipc import log
        log(f"Could not write scenes.json: {exc}")


def _detect_ipc(
    video: Path,
    output: Optional[Path],
    method: str,
    min_duration: float,
    workers: int,
    similarity_threshold: float,
    edge_threshold: float,
    edge_radius: float,
    ai_threshold: float = 0.5,
    decode_method: str = "ffmpeg",
) -> None:
    import sys
    from ...core.infra.ipc import emit_progress, emit_event, log
    from ...core.detection.keyframe import detect_cuts_by_keyframe
    from ...core.detection.edge import detect_cuts_by_edge
    from ...core.cutting.segmenter import run_ffmpeg_segment, collect_scenes
    from ...core.video import get_video_duration
    from ...core.thumbnails.thumbnails_streaming import generate_thumbnails_streaming

    video_path = str(video.resolve())
    video_stem = video.stem

    if output is None:
        output_dir = str(video.parent / f"{video_stem}_scenes")
    else:
        output_dir = str(output.resolve())

    os.makedirs(output_dir, exist_ok=True)

    emit_progress(10, "Extracting keyframes...")

    def kf_cb(pct: int, msg: str) -> None:
        emit_progress(pct, msg)

    if method == "transnetv2":
        from ...core.detection.ai_scene_detection import TRANSNET_AVAILABLE
        if not TRANSNET_AVAILABLE:
            fail("transnetv2_pytorch not installed. Run: pip install amverge[ml]")
            raise typer.Exit(1)

        from concurrent.futures import ThreadPoolExecutor

        import torch
        from ...core.detection.ai_scene_detection import (
            decode_and_detect_scenes,
            decode_video_frames_nelux,
            run_model_one_pass,
        )
        from ...core.keyframes.keyframe_align import get_keyframe_timestamps_pyav, classify_scenes_by_keyframe_alignment
        from ...core.codec.codec_utils import check_if_hevc
        from ...core.video.scene_utils import scenes_to_objects
        from ...core.cutting.smart_cut import cut_all_scenes
        from ...core.thumbnails import make_thumbnail

        # `decode_method` has already been downgraded to ffmpeg by the caller if
        # Nelux is not importable, so reaching here with "nelux" means it loaded
        decode_label = "Nelux" if decode_method == "nelux" else "FFmpeg"
        emit_progress(0, f"Starting TransNetV2 detection ({decode_label} decode)...")

        scenes_secs = scenes_frames = None
        if decode_method == "nelux":
            try:
                frames = decode_video_frames_nelux(video_path)
                scenes_secs, scenes_frames = run_model_one_pass(
                    frames, video_path, threshold=ai_threshold
                )
            except Exception as exc:
                # NVDEC turns down files it cannot handle in hardware, 10-bit
                # H.264 above all. a property of this file, not the machine
                log(f"NVDEC cannot decode this file ({exc}), using FFmpeg")
                emit_progress(0, "Starting TransNetV2 detection (FFmpeg decode)...")

        if scenes_secs is None:
            scenes_secs, scenes_frames = decode_and_detect_scenes(video_path, threshold=ai_threshold)

        emit_progress(80, "Extracting keyframe timestamps...")
        keyframes = get_keyframe_timestamps_pyav(video_path)
        is_hevc = check_if_hevc(video_path)

        raw_scenes = scenes_to_objects(scenes_secs=scenes_secs, scenes_frames=scenes_frames)
        scene_pairs = [(s["start_sec"], s["end_sec"]) for s in raw_scenes]
        copy_candidates, reencode_candidates = classify_scenes_by_keyframe_alignment(
            scene_pairs, keyframes
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        scenes_out_dir = Path(output_dir) / "scenes"
        scenes_out_dir.mkdir(parents=True, exist_ok=True)

        copy_idx = {c["scene_id"] for c in copy_candidates}
        phase1_scenes = [s for s in raw_scenes if s["scene_index"] in copy_idx]
        phase2_scenes = [s for s in raw_scenes if s["scene_index"] not in copy_idx]

        def _thumb_path(scene_index: int) -> str:
            return os.path.join(output_dir, f"{video_stem}_{scene_index:04d}.jpg")

        # lets a viewer lay out its grid before any clip is cut; filled in later
        emit_event("INITIAL_CLIPS_READY|" + json.dumps([
            {
                "scene_index": s["scene_index"],
                "start": s["start_sec"],
                "end": s["end_sec"],
                "duration": s["duration_sec"],
                "path": "",
                "thumbnail": _thumb_path(s["scene_index"]),
                "thumbnail_ready": False,
                "original_file": video.name,
            }
            for s in raw_scenes
        ]))

        cut_by_idx: dict[int, dict] = {}

        # per clip as it lands: one pass at the end leaves the grid posterless
        thumb_pool = ThreadPoolExecutor(max_workers=4)
        thumb_futures: list = []

        def _gen_thumb(scene_index: int, clip_path: str) -> None:
            if make_thumbnail(clip_path, _thumb_path(scene_index)):
                emit_event(f"THUMBNAIL_READY|{scene_index}")

        def _on_clip_ready(result: dict) -> None:
            scene_index = result["scene_index"]
            cut_by_idx[scene_index] = result
            clip_path = result.get("clip_path") or ""
            clip_mode = result.get("clip_mode") or "failed"
            emit_event(f"CLIP_READY|{scene_index}|{clip_path}|{clip_mode}")
            if clip_path and os.path.exists(clip_path):
                thumb_futures.append(thumb_pool.submit(_gen_thumb, scene_index, clip_path))

        emit_progress(82, f"Cutting {len(phase1_scenes)} scenes (lossless copy)...")
        cut_all_scenes(
            input_file=video,
            scenes=phase1_scenes,
            keyframes=keyframes,
            out_dir=scenes_out_dir,
            use_cuda=(device == "cuda"),
            is_hevc=is_hevc,
            max_workers=8,
            on_ready=_on_clip_ready,
        )

        emit_event("PHASE1_COMPLETE")

        phase2_total = len(phase2_scenes)
        phase2_done = 0

        if phase2_total:
            emit_event(f"REENCODE_PROGRESS|0|{phase2_total}")

            def _on_reencode_ready(result: dict) -> None:
                nonlocal phase2_done
                _on_clip_ready(result)
                phase2_done += 1
                emit_event(f"REENCODE_PROGRESS|{phase2_done}|{phase2_total}")

            emit_progress(90, f"Cutting {phase2_total} scenes (re-encode)...")
            cut_all_scenes(
                input_file=video,
                scenes=phase2_scenes,
                keyframes=keyframes,
                out_dir=scenes_out_dir,
                use_cuda=(device == "cuda"),
                is_hevc=is_hevc,
                max_workers=2,
                on_ready=_on_reencode_ready,
                emit_progress_updates=False,
            )

        emit_progress(95, "Finishing thumbnails...")
        for future in thumb_futures:
            future.result()
        thumb_pool.shutdown(wait=True)

        scenes = []
        for s in raw_scenes:
            idx = s["scene_index"]
            cut = cut_by_idx.get(idx, {})
            scenes.append({
                "scene_index": idx,
                "start": s["start_sec"],
                "end": s["end_sec"],
                "duration": s["duration_sec"],
                "path": cut.get("clip_path", ""),
                "thumbnail": _thumb_path(idx),
                "original_file": video.name,
            })

        _write_scenes_json(output_dir, scenes)
        emit_progress(100, "Done")
        print(json.dumps(scenes), flush=True)
        return

    if method == "keyframe":
        cut_points = detect_cuts_by_keyframe(video_path, min_duration=min_duration, progress_cb=kf_cb)
    else:
        cut_points = detect_cuts_by_edge(
            video_path,
            threshold=edge_threshold,
            radius=edge_radius,
            min_duration=min_duration,
            progress_cb=kf_cb,
        )

    emit_progress(50, f"Cutting {len(cut_points)} scenes...")

    seg_stem = video_stem.replace("%", "%%")
    output_pattern = os.path.join(output_dir, f"{seg_stem}_%04d.mp4")
    run_ffmpeg_segment(video_path, output_pattern, cut_points)

    total_duration = get_video_duration(video_path)
    scenes = collect_scenes(output_dir, video_stem, cut_points, total_duration)

    emit_progress(75, "Building scenes...")
    emit_progress(90, f"Generating thumbnails for {len(scenes)} scenes...")

    generate_thumbnails_streaming(output_dir, scenes, video_stem)

    _write_scenes_json(output_dir, scenes)
    emit_progress(100, "Done")
    print(json.dumps(scenes), flush=True)
