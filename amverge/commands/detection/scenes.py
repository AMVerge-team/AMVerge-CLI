from __future__ import annotations

import json
from pathlib import Path

import typer

from ...ui import banner, console, make_table, fail


def scenes(
    video: Path = typer.Argument(..., help="Video file", exists=True),
    cache_dir: Path = typer.Option(None, "--cache-dir", help="Directory containing .npy cache (default: next to video)"),
    as_json: bool = typer.Option(False, "--json", help="Output raw JSON"),
    min_duration: float = typer.Option(0.0, "--min-duration", help="Filter scenes shorter than N seconds"),
) -> None:
    """Show detected scenes from a TransNetV2 cache."""
    from ...core.infra.ipc import build_video_cache_prefix
    from ...core.video.scene_utils import scenes_to_objects

    import numpy as np

    banner("scenes")

    video = video.resolve()
    search_dir = cache_dir.resolve() if cache_dir else video.parent

    prefix = build_video_cache_prefix(video)
    secs_path = search_dir / f"{prefix}_secs.npy"
    frames_path = search_dir / f"{prefix}_frames.npy"

    if not secs_path.exists() or not frames_path.exists():
        fail(f"No scene cache for {video.name}")
        fail(f"Expected: {secs_path}")
        fail("Run detection first or check --cache-dir")
        raise typer.Exit(1)

    scenes_secs = np.load(secs_path)
    scenes_frames = np.load(frames_path)
    scene_list = scenes_to_objects(scenes_secs, scenes_frames)

    if min_duration > 0:
        scene_list = [s for s in scene_list if s["duration_sec"] >= min_duration]

    if as_json:
        print(json.dumps(scene_list))
        return

    t = make_table(
        ("#",        "muted",   {"justify": "right", "width": 6}),
        ("Start",    "label",   {"justify": "right", "width": 12}),
        ("End",      "label",   {"justify": "right", "width": 12}),
        ("Duration", "accent",  {"justify": "right", "width": 12}),
        ("Frames",   "muted",   {"justify": "right", "width": 14}),
        title=f"{video.name} - {len(scene_list)} scenes",
    )

    total_duration = 0.0
    for s in scene_list:
        dur = s["duration_sec"]
        total_duration += dur
        frames_range = f"{s['start_frame']}-{s['end_frame']}"
        t.add_row(
            str(s["scene_index"]),
            f"{s['start_sec']:.3f}s",
            f"{s['end_sec']:.3f}s",
            f"{dur:.3f}s",
            frames_range,
        )

    console.print(t)
    console.print(
        f"[muted]{len(scene_list)} scenes  ·  "
        f"{total_duration:.2f}s total  ·  "
        f"cache: {prefix}[/]"
    )
