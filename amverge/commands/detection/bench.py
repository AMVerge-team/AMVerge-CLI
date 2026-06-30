from __future__ import annotations

import time
from pathlib import Path

import typer

from ...ui import banner, console, make_table, make_progress, warn


def bench(
    video: Path = typer.Argument(..., help="Video file to benchmark", exists=True),
    skip_ml: bool = typer.Option(False, "--skip-ml", help="Skip TransNetV2 benchmark"),
) -> None:
    """Benchmark keyframe scan and TransNetV2 inference on a video."""
    banner("bench")

    video = video.resolve()
    results: list[tuple[str, str, str]] = []

    # Keyframe scan
    with make_progress(transient=True) as progress:
        progress.add_task("Scanning keyframes...", total=None)
        t0 = time.perf_counter()
        from ...core.keyframes.keyframe_align import get_keyframe_timestamps_pyav
        kf = get_keyframe_timestamps_pyav(str(video))
        kf_elapsed = time.perf_counter() - t0

    results.append(("keyframe scan", f"{kf_elapsed:.2f}s", f"{len(kf):,} keyframes"))

    # ffprobe duration
    with make_progress(transient=True) as progress:
        progress.add_task("Probing video...", total=None)
        t0 = time.perf_counter()
        from ...core.video.probe_utils import probe_video_duration, probe_video_fps
        duration = probe_video_duration(video)
        fps = probe_video_fps(video)
        probe_elapsed = time.perf_counter() - t0

    total_frames = int(duration * fps)
    results.append(("ffprobe", f"{probe_elapsed:.2f}s", f"{duration:.1f}s video  {fps:.3f} fps  ~{total_frames:,} frames"))

    # TransNetV2
    if not skip_ml:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

            with make_progress(transient=True) as progress:
                progress.add_task(f"TransNetV2 decode ({device})...", total=None)
                t0 = time.perf_counter()
                from ...core.detection.ai_scene_detection import decode_video_frames_nelux
                try:
                    frames = decode_video_frames_nelux(video)
                    decode_elapsed = time.perf_counter() - t0
                    n_frames = len(frames)
                    results.append((
                        f"nelux decode ({device})",
                        f"{decode_elapsed:.2f}s",
                        f"{n_frames:,} frames  {n_frames / decode_elapsed:.0f} fps throughput",
                    ))

                    with make_progress(transient=True) as progress2:
                        progress2.add_task(f"TransNetV2 inference ({device})...", total=None)
                        t0 = time.perf_counter()
                        from ...core.detection.ai_scene_detection import run_model_one_pass
                        scenes_secs, scenes_frames = run_model_one_pass(frames, video)
                        infer_elapsed = time.perf_counter() - t0

                    results.append((
                        f"TransNetV2 inference ({device})",
                        f"{infer_elapsed:.2f}s",
                        f"{len(scenes_secs):,} scenes detected  {n_frames / infer_elapsed:.0f} fps throughput",
                    ))

                except ImportError:
                    t0 = time.perf_counter()
                    from ...core.detection.ai_scene_detection import decode_and_detect_scenes
                    scenes_secs, _ = decode_and_detect_scenes(video)
                    total_elapsed = time.perf_counter() - t0
                    results.append((
                        f"TransNetV2 full ({device})",
                        f"{total_elapsed:.2f}s",
                        f"{len(scenes_secs):,} scenes detected",
                    ))

        except ImportError:
            warn("torch not installed - skipping ML benchmark  (pip install amverge[ml])")

    t = make_table(
        ("Stage",       "#22c55e bold", {"no_wrap": True, "width": 30}),
        ("Time",        "label",        {"justify": "right", "width": 10}),
        ("Detail",      "muted",        {}),
        title=f"{video.name}",
    )
    for stage, elapsed, detail in results:
        t.add_row(stage, elapsed, detail)

    console.print(t)
