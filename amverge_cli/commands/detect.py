from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from ..pipeline import detect_scenes, DetectResult

console = Console()
err = Console(stderr=True)

_STAGE_LABELS = {
    "detect": "Detecting cuts",
    "segment": "Cutting scenes",
    "thumbnails": "Thumbnails",
    "similarity": "Similarity",
}


def detect(
    video: Path = typer.Argument(..., help="Input video file", exists=True),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    method: str = typer.Option("keyframe", "--method", "-m", help="Detection method: keyframe, edge"),
    format: str = typer.Option("table", "--format", "-f", help="Output format: table, json, paths"),
    json_output: Optional[Path] = typer.Option(None, "--json-output", help="Also save JSON to this path"),
    no_thumbnails: bool = typer.Option(False, "--no-thumbnails", help="Skip thumbnail generation"),
    no_similarity: bool = typer.Option(False, "--no-similarity", help="Skip similarity check"),
    min_duration: float = typer.Option(0.25, "--min-duration", help="Min scene duration in seconds"),
    workers: int = typer.Option(4, "--workers", help="Thumbnail worker threads"),
    similarity_threshold: float = typer.Option(0.10, "--similarity-threshold", help="Similarity threshold"),
    edge_threshold: float = typer.Option(0.15, "--edge-threshold", help="Edge detection threshold"),
    edge_radius: float = typer.Option(0.6, "--edge-radius", help="Keyframe window radius for edge detection"),
) -> None:
    """Split a video into scenes at cut boundaries."""
    fmt = format.lower()
    if fmt not in ("table", "json", "paths"):
        err.print("[red]--format must be: table, json, or paths")
        raise typer.Exit(1)

    if method not in ("keyframe", "edge"):
        err.print("[red]--method must be: keyframe or edge")
        raise typer.Exit(1)

    current_task_id = None

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[cyan]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=err,
        transient=True,
    ) as progress:
        tasks: dict[str, object] = {}

        def on_progress(stage: str, pct: int, msg: str) -> None:
            label = _STAGE_LABELS.get(stage, stage)
            if stage not in tasks:
                tasks[stage] = progress.add_task(f"{label}...", total=100)
            tid = tasks[stage]
            progress.update(tid, completed=pct, description=f"{label}: {msg[:50]}" if msg else label)

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
        err.print("[red]No scenes detected.")
        raise typer.Exit(1)

    if json_output:
        json_output.write_text(json.dumps(result.to_dict(), indent=2))
        err.print(f"[green]JSON saved → {json_output}")

    similar_set = {idx for pair in result.similar_pairs for idx in pair}

    if fmt == "json":
        console.print_json(json.dumps(result.to_dict()))
    elif fmt == "paths":
        for scene in result.scenes:
            console.print(scene.path)
    else:
        table = Table(
            title=f"{video.stem} — {len(result.scenes)} scenes  [{method}]",
            show_lines=False,
        )
        table.add_column("#", style="dim", justify="right", width=5)
        table.add_column("Start", justify="right", width=9)
        table.add_column("End", justify="right", width=9)
        table.add_column("Duration", justify="right", width=9)
        table.add_column("~", justify="center", width=3)

        for scene in result.scenes:
            table.add_row(
                str(scene.index),
                f"{scene.start:.2f}s",
                f"{scene.end:.2f}s",
                f"{scene.duration:.2f}s",
                "[yellow]~[/yellow]" if scene.index in similar_set else "",
            )

        console.print(table)
        console.print(f"[dim]Scenes JSON → {result.scenes_json}[/dim]")
