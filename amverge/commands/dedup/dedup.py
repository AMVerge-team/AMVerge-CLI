from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ...ui import banner, console, make_progress, ok, fail
from ...core.dedup import DEDUP_METHODS

_DEFAULT_THRESHOLD = {"ffmpeg": 0.33, "ssim": 0.987, "framediff": 10.0}


def dedup(
    input: Optional[Path] = typer.Argument(None, help="Input video file"),
    output: Path = typer.Option(None, "--output", "-o", help="Output video file"),
    method: str = typer.Option("ffmpeg", "--method", "-m", help="Dedup method: ffmpeg, ssim, framediff"),
    threshold: Optional[float] = typer.Option(None, "--threshold", "-t", help="Detection threshold (method-specific; sensible default per method)"),
    min_change_pct: float = typer.Option(2.0, "--min-change-pct", help="Min changed pixel %% for framediff method"),
    list_methods: bool = typer.Option(False, "--list-methods", help="List available dedup methods"),
) -> None:
    """Remove duplicate / dead frames from a video.

    Methods: ffmpeg (mpdecimate, no deps), ssim (OpenCV, quality-aware),
    framediff (OpenCV, pixel motion). Output preserves audio, color and bit depth.
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

    if output is None:
        output = input.parent / f"{input.stem}_deduped{input.suffix}"

    if threshold is None:
        threshold = _DEFAULT_THRESHOLD.get(method, 0.0)

    entry = DEDUP_METHODS[method]

    banner("dedup")
    console.print(f"  Method: [accent]{entry['name']}[/accent]")
    console.print(f"  Threshold: [accent]{threshold}[/accent]")
    if method == "framediff":
        console.print(f"  Min change: [accent]{min_change_pct}%[/accent]")
    console.print(f"  Input:  [dim]{input}[/dim]")
    console.print(f"  Output: [dim]{output}[/dim]")

    stats = None
    with make_progress() as progress:
        task_id = progress.add_task("Dedup...", total=100)

        def _progress_cb(pct, msg):
            progress.update(task_id, completed=pct, description=msg)

        try:
            if method == "ffmpeg":
                from ...core.dedup import dedup_ffmpeg
                _, stats = dedup_ffmpeg(str(input.resolve()), str(output.resolve()), threshold, _progress_cb)
            elif method == "ssim":
                from ...core.dedup import dedup_ssim, SSIM_AVAILABLE
                if not SSIM_AVAILABLE:
                    fail("SSIM method requires opencv. Run: pip install amverge[dedup]")
                    raise typer.Exit(1)
                _, stats = dedup_ssim(str(input.resolve()), str(output.resolve()), threshold, _progress_cb)
            elif method == "framediff":
                from ...core.dedup import dedup_framediff, FRAMEDIFF_AVAILABLE
                if not FRAMEDIFF_AVAILABLE:
                    fail("FrameDiff method requires opencv. Run: pip install amverge[dedup]")
                    raise typer.Exit(1)
                _, stats = dedup_framediff(str(input.resolve()), str(output.resolve()), threshold, min_change_pct, _progress_cb)
        except typer.Exit:
            raise
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
    ok(f"Saved: {output}")
