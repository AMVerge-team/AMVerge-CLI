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
    if decode_method != "ffmpeg" and method != "transnetv2":
        warn("--decode-method only applies to --method transnetv2; ignoring")
        decode_method = "ffmpeg"
    if method == "transnetv2" and decode_method == "nelux":
        from ...core.detection.nelux_runtime import nelux_available
        if not nelux_available():
            warn("Nelux unavailable, falling back to FFmpeg parallel decode")
            decode_method = "ffmpeg"

    if ipc:
        _detect_ipc(video, output, method, min_duration, workers, similarity_threshold, edge_threshold, edge_radius)
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


def _detect_ipc(
    video: Path,
    output: Optional[Path],
    method: str,
    min_duration: float,
    workers: int,
    similarity_threshold: float,
    edge_threshold: float,
    edge_radius: float,
) -> None:
    import sys
    from ...core.infra.ipc import emit_progress, emit_event
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

        import torch
        from ...core.detection.ai_scene_detection import decode_and_detect_scenes
        from ...core.keyframes.keyframe_align import get_keyframe_timestamps_pyav, classify_scenes_by_keyframe_alignment
        from ...core.codec.codec_utils import check_if_hevc
        from ...core.video.scene_utils import scenes_to_objects
        from ...core.cutting.smart_cut import cut_all_scenes

        emit_progress(0, "Starting TransNetV2 detection...")
        scenes_secs, scenes_frames = decode_and_detect_scenes(video_path)

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

        cut_by_idx: dict[int, dict] = {}

        def _on_clip_ready(result: dict) -> None:
            cut_by_idx[result["scene_index"]] = result

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

        if phase2_scenes:
            emit_progress(90, f"Cutting {len(phase2_scenes)} scenes (re-encode)...")
            cut_all_scenes(
                input_file=video,
                scenes=phase2_scenes,
                keyframes=keyframes,
                out_dir=scenes_out_dir,
                use_cuda=(device == "cuda"),
                is_hevc=is_hevc,
                max_workers=2,
                on_ready=_on_clip_ready,
                emit_progress_updates=False,
            )

        emit_progress(95, "Generating thumbnails...")
        from ...core.thumbnails import make_thumbnail
        for scene in raw_scenes:
            idx = scene["scene_index"]
            thumb_path = os.path.join(output_dir, f"{video_stem}_{idx:04d}.jpg")
            clip_path = cut_by_idx.get(idx, {}).get("clip_path", "")
            if clip_path and os.path.exists(clip_path):
                make_thumbnail(clip_path, thumb_path)

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
                "thumbnail": os.path.join(output_dir, f"{video_stem}_{idx:04d}.jpg"),
                "original_file": video.name,
            })

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

    emit_progress(100, "Done")
    print(json.dumps(scenes), flush=True)
