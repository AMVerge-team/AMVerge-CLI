from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ...ui import banner, console, make_progress, ok, fail
from ...core.dedup import DEDUP_METHODS, run_dedup
from ...core.dedup.dispatch import DEFAULT_THRESHOLD


def dedup(
    input: Optional[Path] = typer.Argument(None, help="Input video file"),
    output: Path = typer.Option(None, "--output", "-o", help="Output video file"),
    method: str = typer.Option("ffmpeg", "--method", "-m", help="Dedup method: ffmpeg, ssim, framediff, advanced"),
    threshold: Optional[float] = typer.Option(None, "--threshold", "-t", help="Detection threshold (method-specific; sensible default per method). For 'advanced' this is sensitivity."),
    min_change_pct: float = typer.Option(2.0, "--min-change-pct", help="Min changed pixel %% for framediff method"),
    codec: Optional[str] = typer.Option(None, "--codec", "-c", help="Output codec profile (e.g. h264_high, h265_main10, prores_422). Default x264."),
    crf: int = typer.Option(18, "--crf", help="Encode quality, lower = better (ignored for prores)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Analyze only, report removals, write no output (ssim/framediff/advanced)"),
    export_frames: Optional[Path] = typer.Option(None, "--export-frames", help="Write kept/removed frame ranges to a CSV (ssim/framediff/advanced)"),
    list_methods: bool = typer.Option(False, "--list-methods", help="List available dedup methods"),
) -> None:
    """Remove duplicate / dead frames from a video.

    Methods: ffmpeg (mpdecimate, no deps), ssim (OpenCV, quality-aware),
    framediff (OpenCV, pixel motion), advanced (OpenCV, region grid + optical
    flow + edges + cadence). Output preserves audio, color and bit depth.
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

    if method not in DEDUP_METHODS:
        fail(f"Unknown method '{method}'. Valid: {', '.join(DEDUP_METHODS.keys())}")
        raise typer.Exit(1)

    if (dry_run or export_frames) and method == "ffmpeg":
        fail("--dry-run and --export-frames need ssim, framediff or advanced (ffmpeg can't enumerate frames).")
        raise typer.Exit(1)

    if output is None:
        output = input.parent / f"{input.stem}_deduped{input.suffix}"

    if threshold is None:
        threshold = DEFAULT_THRESHOLD.get(method, 0.0)

    entry = DEDUP_METHODS[method]

    banner("dedup")
    console.print(f"  Method: [accent]{entry['name']}[/accent]")
    console.print(f"  {'Sensitivity' if method == 'advanced' else 'Threshold'}: [accent]{threshold}[/accent]")
    if method == "framediff":
        console.print(f"  Min change: [accent]{min_change_pct}%[/accent]")
    console.print(f"  Codec: [accent]{codec or 'x264 (default)'}[/accent]")
    console.print(f"  Input:  [dim]{input}[/dim]")
    if dry_run:
        console.print("  Mode:   [warn]dry run (no output)[/warn]")
    else:
        console.print(f"  Output: [dim]{output}[/dim]")
    if export_frames:
        console.print(f"  Frames CSV: [dim]{export_frames}[/dim]")

    stats = None
    with make_progress() as progress:
        task_id = progress.add_task("Dedup...", total=100)

        def _progress_cb(pct, msg):
            progress.update(task_id, completed=pct, description=msg)

        try:
            _, stats = run_dedup(
                str(input.resolve()),
                str(output.resolve()),
                method=method,
                threshold=threshold,
                min_change_pct=min_change_pct,
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
                f"  Cadence: every [accent]{stats['cadence']}[/accent] frames "
                f"(confidence [accent]{stats['confidence']}[/accent])"
            )
    if export_frames:
        ok(f"Frame list: {export_frames}")
    if dry_run:
        ok("Dry run complete (no output written)")
    else:
        ok(f"Saved: {output}")
