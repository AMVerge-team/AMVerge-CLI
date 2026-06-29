from __future__ import annotations

import json
from pathlib import Path

import typer

from ...ui import banner, console, make_table, make_progress


def keyframes(
    video: Path = typer.Argument(..., help="Video file", exists=True),
    as_json: bool = typer.Option(False, "--json", help="Output raw JSON list"),
    count_only: bool = typer.Option(False, "--count", help="Print count only"),
    threshold: float = typer.Option(0.2, "--threshold", help="Snap threshold in seconds (for cut mode column)"),
) -> None:
    """List all keyframe timestamps for a video."""
    from ...core.keyframes.keyframe_align import get_keyframe_timestamps_pyav

    banner("keyframes")

    video = video.resolve()

    with make_progress(transient=True) as progress:
        progress.add_task(f"Scanning {video.name}...", total=None)
        kf = get_keyframe_timestamps_pyav(str(video))

    if count_only:
        console.print(str(len(kf)))
        return

    if as_json:
        print(json.dumps(kf))
        return

    if not kf:
        console.print("[muted]No keyframes found.[/]")
        return

    t = make_table(
        ("#",         "muted",  {"justify": "right", "width": 6}),
        ("Time (s)",  "label",  {"justify": "right", "width": 12}),
        ("Gap",       "muted",  {"justify": "right", "width": 10}),
        title=f"{video.name} - {len(kf):,} keyframes",
    )

    prev = None
    for i, ts in enumerate(kf):
        gap = f"{ts - prev:.3f}s" if prev is not None else "-"
        t.add_row(str(i + 1), f"{ts:.3f}", gap)
        prev = ts

    console.print(t)

    if len(kf) >= 2:
        avg = (kf[-1] - kf[0]) / (len(kf) - 1)
        console.print(f"[muted]Avg. keyframe gap: {avg:.2f}s[/]")
