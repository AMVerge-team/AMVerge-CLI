from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ...ui import banner, console, make_progress, ok, fail
from ...core.dedup import (
    DEDUP_METHODS,
    run_dedup_simple,
    auto_detect_method,
    PRESET_LABELS,
)


def dedup(
    input: Optional[Path] = typer.Argument(None, help="Input video file"),
    output: Path = typer.Option(None, "--output", "-o", help="Output video file"),
    aggressive: bool = typer.Option(False, "--aggressive", help="Remove more frames (best for clean animation)"),
    gentle: bool = typer.Option(False, "--gentle", help="Remove fewer frames, safest (best for grainy/live-action)"),
    codec: Optional[str] = typer.Option(None, "--codec", "-c", help="Output codec (e.g. h264_high, h265_main10). Default x264."),
    crf: int = typer.Option(18, "--crf", help="Quality (lower = better)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview what would be removed, write no output"),
    export_frames: Optional[Path] = typer.Option(None, "--export-frames", help="Write kept/removed frame ranges to CSV"),
    list_methods: bool = typer.Option(False, "--list-methods", help="List available dedup methods"),
) -> None:
    """Remove duplicate frames from video.

    Auto-detects the best method and applies sensible presets.
    No configuration needed — just point at a video.

    Presets:
      (default)  Balanced
      --aggressive  Removes more, slightly riskier (clean animation)
      --gentle      Removes fewer, safest (grainy or live-action)

    Examples:
      amverge dedup video.mp4
      amverge dedup video.mp4 --aggressive
      amverge dedup video.mp4 --gentle --dry-run
    """
    if list_methods:
        banner("dedup methods")
        console.print()
        for key, entry in DEDUP_METHODS.items():
            req = entry.get("requires") or "none"
            console.print(f"  [accent]{key}[/accent] - {entry['name']}")
            console.print(f"    {entry['description']}")
            console.print(f"    Requires: [dim]{req}[/]")
        console.print()
        return

    if input is None:
        fail("Missing input video. Pass a file, or use --list-methods.")
        raise typer.Exit(1)

    if not input.exists():
        fail(f"File not found: {input}")
        raise typer.Exit(1)

    if output is None:
        suffix = "_dry" if dry_run else "_deduped"
        output = input.parent / f"{input.stem}{suffix}{input.suffix}"

    if aggressive and gentle:
        fail("Pick --aggressive or --gentle, not both.")
        raise typer.Exit(1)

    if aggressive:
        preset = "aggressive"
    elif gentle:
        preset = "gentle"
    else:
        preset = "normal"

    method = auto_detect_method()
    method_name = DEDUP_METHODS[method]["name"]

    banner("dedup")
    console.print(f"  Method: [accent]{method_name}[/accent] (auto-detected)")
    console.print(f"  Preset: [accent]{preset}[/accent] — {PRESET_LABELS[preset]}")
    if codec:
        console.print(f"  Codec: [accent]{codec}[/accent]")
    console.print(f"  Input:  [dim]{input}[/dim]")
    if dry_run:
        console.print("  Mode:   [warn]dry run (no output)[/warn]")
    else:
        console.print(f"  Output: [dim]{output}[/dim]")
    if export_frames:
        console.print(f"  Frames CSV: [dim]{export_frames}[/dim]")

    if (dry_run or export_frames) and method == "ffmpeg":
        fail("Dry-run and frame export require OpenCV.\n  pip install amverge[dedup]")
        raise typer.Exit(1)

    stats = None
    with make_progress() as progress:
        task_id = progress.add_task("Dedup...", total=100)

        def _progress_cb(pct, msg):
            progress.update(task_id, completed=pct, description=msg)

        try:
            _, stats = run_dedup_simple(
                str(input.resolve()),
                str(output.resolve()),
                preset=preset,
                codec=codec,
                crf=crf,
                dry_run=dry_run,
                export_frames=str(export_frames.resolve()) if export_frames else None,
                progress_cb=_progress_cb,
            )
        except Exception as e:
            fail(str(e))
            raise typer.Exit(1)

    if stats:
        console.print(
            f"  Frames: [accent]{stats['frames_in']}[/accent] -> "
            f"[accent]{stats['frames_out']}[/accent] "
            f"([accent]{stats['frames_removed']}[/accent] removed, "
            f"[accent]{stats['pct_removed']}%[/accent])"
        )
        if "cadence" in stats and stats["cadence"]:
            console.print(
                f"  Animation cadence: every [accent]{stats['cadence']}[/accent] frames "
                f"(confidence [accent]{stats['confidence']}[/accent])"
            )
    if export_frames:
        ok(f"Frame list: {export_frames}")
    if dry_run:
        ok("Dry run complete (no output written)")
    else:
        ok(f"Saved: {output}")
