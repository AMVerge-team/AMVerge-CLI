from __future__ import annotations

import json
import uuid
from pathlib import Path

import typer

from ...core.infra.ipc import emit_progress, emit_event, log, check_if_path_exists, build_video_cache_prefix
from ...core.thumbnails import make_thumbnail
from ...core.detection.ai_scene_detection import decode_video_frames_nelux, run_model_one_pass
from ...core.video.probe_utils import probe_video_duration, probe_video_fps, probe_video_dimensions
from ...core.video.scene_utils import scenes_to_objects
from ...core.keyframes.keyframe_align import get_keyframe_timestamps_pyav, classify_scenes_by_keyframe_alignment
from ...core.codec.codec_utils import check_if_hevc
from ...core.cutting.smart_cut import cut_all_scenes


def backend(
    video_path: str = typer.Argument(..., help="Input video file"),
    output_dir: str = typer.Argument(..., help="Output directory for scene data"),
    scene_detection_method: str = typer.Argument("transnetv2_gpu", hidden=True),
    import_method: str = typer.Argument("video_files", hidden=True),
) -> None:
    """
    Called by Rust exactly like the original backend:
        amverge backend <video_path> <output_dir> <scene_detection_method> <import_method>

    ``scene_detection_method`` selects the device for TransNetV2 inference:
    a ``*_cpu`` value forces CPU, anything else (e.g. ``transnetv2_gpu``) uses
    CUDA when available and falls back to CPU otherwise.

    Emits IPC events to stderr and final JSON to stdout.
    """
    input_video = Path(video_path)
    out_dir = Path(output_dir)

    import numpy as np
    try:
        import torch
    except ImportError:
        print(f"Install with: pip install amverge[ml]")
        raise SystemExit(1)

    force_cpu = scene_detection_method.lower().endswith("_cpu")
    device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    def _error_exit(error: Exception) -> None:
        import traceback
        log(f"FATAL ERROR: {error}")
        log(traceback.format_exc())
        print(
            json.dumps({
                "schema_version": "1.0",
                "run_id": str(uuid.uuid4()),
                "video": None,
                "cache": None,
                "scenes": [],
                "scenes_secs": [],
                "scenes_frames": [],
                "detector": {"method": "run_model_one_pass", "device": device},
                "warnings": [],
                "error": {"message": str(error), "type": type(error).__name__},
            }),
            flush=True,
        )
        raise typer.Exit(1)

    try:
        emit_progress(0, "Loading video...")
        check_if_path_exists(str(input_video))
        out_dir.mkdir(parents=True, exist_ok=True)

        cache_prefix = build_video_cache_prefix(input_video)
        scenes_secs_path = out_dir / f"{cache_prefix}_secs.npy"
        scenes_frames_path = out_dir / f"{cache_prefix}_frames.npy"
        cache_hit = False

        emit_progress(5, "Preparing scene detection cache...")

        scenes_secs: np.ndarray
        scenes_frames: np.ndarray

        if scenes_secs_path.exists() and scenes_frames_path.exists():
            cache_hit = True
            scenes_secs = np.load(scenes_secs_path)
            scenes_frames = np.load(scenes_frames_path)
            emit_progress(20, "Loaded cached scene detection results.")
        else:
            emit_progress(20, "Decoding frames for TransNetV2...")
            frames = decode_video_frames_nelux(input_video)

            emit_progress(55, "Running TransNetV2 scene detection...")
            scenes_secs, scenes_frames = run_model_one_pass(frames, input_video)

            np.save(scenes_secs_path, scenes_secs)
            np.save(scenes_frames_path, scenes_frames)
            emit_progress(80, "Saved scene detection cache.")

        input_video_duration = probe_video_duration(input_video)
        input_video_fps = probe_video_fps(input_video)
        input_video_width, input_video_height = probe_video_dimensions(input_video)
        scenes = scenes_to_objects(scenes_secs=scenes_secs, scenes_frames=scenes_frames)

        if import_method == "video_files":
            source_str = str(input_video)
            source_name = input_video.name

            scenes_out_dir = out_dir / "scenes"

            def _poster_path(scene_index: int) -> Path:
                return scenes_out_dir / f"scene_{scene_index:04d}.jpg"

            initial_clips = [
                {
                    "scene_index": s["scene_index"],
                    "start_sec": s["start_sec"],
                    "end_sec": s["end_sec"],
                    "path": source_str,
                    "thumbnail": str(_poster_path(s["scene_index"])),
                    "thumbnail_ready": False,
                    "original_file": source_name,
                    "original_path": source_str,
                    "clip_path": None,
                    "clip_mode": None,
                }
                for s in scenes
            ]
            emit_event(f"INITIAL_CLIPS_READY|{json.dumps(initial_clips)}")

            emit_progress(82, "Extracting keyframe timestamps...")
            keyframes = get_keyframe_timestamps_pyav(str(input_video))
            is_hevc = check_if_hevc(str(input_video))

            scene_pairs = [(s["start_sec"], s["end_sec"]) for s in scenes]
            copy_candidates, reencode_candidates = classify_scenes_by_keyframe_alignment(
                scene_pairs, keyframes
            )
            copy_idx = {c["scene_id"] for c in copy_candidates}
            phase1_scenes = [s for s in scenes if s["scene_index"] in copy_idx]
            phase2_scenes = [s for s in scenes if s["scene_index"] not in copy_idx]
            log(
                f"Video cut split: {len(phase1_scenes)} lossless copies, "
                f"{len(phase2_scenes)} re-encodes"
            )

            cut_by_idx: dict[int, dict] = {}


            import concurrent.futures as _futures

            thumb_pool = _futures.ThreadPoolExecutor(max_workers=4)
            thumb_futures: list = []

            def _gen_thumb(scene_index: int, clip_path: str, is_copy: bool) -> None:
                if make_thumbnail(clip_path, str(_poster_path(scene_index)), first_keyframe=is_copy):
                    emit_event(f"THUMBNAIL_READY|{scene_index}")
                else:
                    log(f"Thumbnail produced no frame for scene {scene_index}")

            def _on_clip_ready(result: dict) -> None:
                scene_index = result["scene_index"]
                cut_by_idx[scene_index] = result
                clip_path = result.get("clip_path") or ""
                clip_mode = result.get("clip_mode") or "failed"
                emit_event(f"CLIP_READY|{scene_index}|{clip_path}|{clip_mode}")
                # Queue the poster off the cut worker threads.
                if clip_path and Path(clip_path).exists():
                    thumb_futures.append(
                        thumb_pool.submit(_gen_thumb, scene_index, clip_path, clip_mode == "copy")
                    )

            cut_all_scenes(
                input_file=input_video,
                scenes=phase1_scenes,
                keyframes=keyframes,
                out_dir=scenes_out_dir,
                use_cuda=(device == "cuda"),
                is_hevc=is_hevc,
                max_workers=8,
                on_ready=_on_clip_ready,
                progress_range=(82, 99),
            )

            emit_progress(100, "Keyframe clips ready")
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

            cut_all_scenes(
                input_file=input_video,
                scenes=phase2_scenes,
                keyframes=keyframes,
                out_dir=scenes_out_dir,
                use_cuda=(device == "cuda"),
                is_hevc=is_hevc,
                max_workers=2,
                on_ready=_on_reencode_ready,
                emit_progress_updates=False,
            )

            # Wait for every queued poster so the final manifest's thumbnail_ready
            # flags reflect which jpgs actually exist on disk.
            for _f in thumb_futures:
                _f.result()
            thumb_pool.shutdown(wait=True)

            for scene in scenes:
                scene_index = scene["scene_index"]
                cut = cut_by_idx.get(scene_index, {})
                scene["clip_path"] = cut.get("clip_path")
                scene["clip_mode"] = cut.get("clip_mode", "failed")
                poster = _poster_path(scene_index)
                scene["thumbnail"] = str(poster)
                scene["thumbnail_ready"] = poster.exists()

        emit_progress(97, "Finalizing scene manifest...")

        result = {
            "schema_version": "1.0",
            "run_id": str(uuid.uuid4()),
            "video": {
                "video_file_path": str(input_video),
                "duration_sec": input_video_duration,
                "width": input_video_width,
                "height": input_video_height,
                "fps": input_video_fps,
            },
            "cache": {
                "cache_hit": cache_hit,
                "secs_path": str(scenes_secs_path),
                "frames_path": str(scenes_frames_path),
            },
            "scenes": scenes,
            "scenes_secs": scenes_secs.tolist(),
            "scenes_frames": scenes_frames.tolist(),
            "detector": {
                "method": "run_model_one_pass",
                "device": device,
            },
            "warnings": [],
            "error": None,
        }
        print(json.dumps(result), flush=True)

    except Exception as error:
        _error_exit(error)
