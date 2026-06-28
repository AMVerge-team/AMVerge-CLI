from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from ..pipeline import detect_scenes, DetectResult
from ..ui import banner, console, err, make_progress, make_table, ok, fail, dim

_STAGE_LABELS = {
    "detect":     "Detecting cuts",
    "segment":    "Cutting scenes",
    "thumbnails": "Thumbnails",
    "similarity": "Similarity check",
}


def detect(
    video: Path = typer.Argument(..., help="Input video file", exists=True),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    method: str = typer.Option("keyframe", "--method", "-m", help="keyframe · edge"),
    format: str = typer.Option("table", "--format", "-f", help="table · json · paths"),
    json_output: Optional[Path] = typer.Option(None, "--json-output", help="Save JSON to file"),
    no_thumbnails: bool = typer.Option(False, "--no-thumbnails"),
    no_similarity: bool = typer.Option(False, "--no-similarity"),
    min_duration: float = typer.Option(0.25, "--min-duration"),
    workers: int = typer.Option(4, "--workers"),
    similarity_threshold: float = typer.Option(0.10, "--similarity-threshold"),
    edge_threshold: float = typer.Option(0.15, "--edge-threshold"),
    edge_radius: float = typer.Option(0.6, "--edge-radius"),
) -> None:
    """Detect scenes in a video file."""
    fmt = format.lower()
    if fmt not in ("table", "json", "paths"):
        fail("--format must be: table, json, or paths")
        raise typer.Exit(1)
    if method not in ("keyframe", "edge"):
        fail("--method must be: keyframe or edge")
        raise typer.Exit(1)

    banner("detect")

    with make_progress() as progress:
        tasks: dict[str, object] = {}

        def on_progress(stage: str, pct: int, msg: str) -> None:
            label = _STAGE_LABELS.get(stage, stage)
            if stage not in tasks:
                tasks[stage] = progress.add_task(label, total=100)
            progress.update(tasks[stage], completed=pct, description=label)

        result: DetectResult = detect_scenes(
            str(video.resolve()),
            output_dir=str(output.resolve()) if output else None,
            method=method,
            min_duration=min_duration,
            thumbnails=not no_thumbnails,
            similarity=not no_similarity and not no_thumbnails,
            similarity_threshold=similarity_threshold,
            thumbnail_workers=workers,
            edge_threshold=edge_threshold,
            edge_radius=edge_radius,
            progress=on_progress,
        )

    if not result.scenes:
        fail("No scenes detected.")
        raise typer.Exit(1)

    if json_output:
        json_output.write_text(json.dumps(result.to_dict(), indent=2))
        ok(f"JSON → {json_output}")

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
    dim(f"scenes.json → {result.scenes_json}")
