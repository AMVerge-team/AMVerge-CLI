from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ...ui import banner, console, err, ok, fail
from ...core.infra.diagnostics import get_gpu_info
from ...core.upscaling.monitor import SystemMonitor, format_eta
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
    anime: bool = typer.Option(False, "--anime", help="Optimized for animation (on-twos/threes cadence, flat colors)"),
    gentle: bool = typer.Option(False, "--gentle", help="Remove fewer frames, safest (best for grainy/live-action)"),
    codec: Optional[str] = typer.Option(None, "--codec", "-c", help="Output codec (e.g. h264_high, h265_main10). Default x264."),
    crf: int = typer.Option(18, "--crf", help="Quality (lower = better)"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview what would be removed, write no output"),
    export_frames: Optional[Path] = typer.Option(None, "--export-frames", help="Write kept/removed frame ranges to CSV"),
    list_methods: bool = typer.Option(False, "--list-methods", help="List available dedup methods"),
    no_monitor: bool = typer.Option(False, "--no-monitor", help="Disable system monitor during processing"),
) -> None:
    """Remove duplicate frames from video.

    Auto-detects the best method and applies sensible presets.
    No configuration needed — just point at a video.

    Presets:
      (default)   Balanced
      --anime       Optimized for animation (on-twos/threes cadence)
      --aggressive  Removes more, slightly riskier (clean animation)
      --gentle      Removes fewer, safest (grainy or live-action)

    Examples:
      amverge dedup video.mp4
      amverge dedup video.mp4 --anime
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

    if sum([aggressive, anime, gentle]) > 1:
        fail("Pick one preset: --aggressive, --anime, or --gentle.")
        raise typer.Exit(1)

    if aggressive:
        preset = "aggressive"
    elif anime:
        preset = "anime"
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

    gpu_info = get_gpu_info()
    if gpu_info.get("cuda_available"):
        vram = gpu_info.get("vram_gb", 0)
        console.print(f"  GPU: [accent]{gpu_info.get('gpu_name', 'GPU')}[/accent]  "
                      f"VRAM: [accent]{vram:.1f} GB[/accent]")

    monitor = SystemMonitor(enabled=not no_monitor)
    monitor.stats["gpu_name"] = gpu_info.get("gpu_name", "GPU")
    monitor.start()

    def _update_display():
        from rich.live import Live
        from rich.panel import Panel
        from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
        from rich.console import Group

        if not hasattr(_update_display, "live"):
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
            )
            _update_display.task = progress.add_task("Dedup...", total=100)
            _update_display.progress = progress
            _update_display.live = Live(progress, console=err, refresh_per_second=4, transient=True)
            _update_display.live.start()

        s = monitor.stats
        _update_display.progress.update(_update_display.task, completed=s["pct"], description=s["msg"])

        if monitor.enabled and hasattr(_update_display, "live"):
            lines = []
            gpu_parts = []
            if s.get("gpu_util") is not None:
                gpu_parts.append(f"GPU {s['gpu_util']:.0f}%")
            if s.get("gpu_temp") is not None:
                gpu_parts.append(f"{s['gpu_temp']:.0f}°C")
            if s.get("vram_used_mb") is not None and s.get("vram_total_mb"):
                gpu_parts.append(f"VRAM {s['vram_used_mb']:.0f}/{s['vram_total_mb']:.0f} MB")
            if gpu_parts:
                lines.append(f"  {s.get('gpu_name', 'GPU')}: {' | '.join(gpu_parts)}")

            cpu_parts = []
            if s.get("cpu_percent") is not None:
                cpu_parts.append(f"CPU {s['cpu_percent']:.0f}%")
            if s.get("ram_used_gb") is not None and s.get("ram_total_gb"):
                cpu_parts.append(f"RAM {s['ram_used_gb']:.1f}/{s['ram_total_gb']:.1f} GB")
            if cpu_parts:
                lines.append(f"  {' | '.join(cpu_parts)}")

            status_parts = []
            if s.get("eta_s") is not None and s["eta_s"] != float("inf"):
                status_parts.append(f"ETA {format_eta(s['eta_s'])}")
            if s.get("elapsed_s"):
                status_parts.append(f"elapsed {format_eta(s['elapsed_s'])}")
            if status_parts:
                lines.append(f"  {' | '.join(status_parts)}")

            content = [_update_display.progress]
            if lines:
                content.append(Panel("\n".join(lines), border_style="#22c55e", padding=(0, 1)))
            _update_display.live.update(Group(*content))

    def _progress_cb(pct, msg):
        monitor.progress_callback(pct, msg)
        _update_display()

    stats = None
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
        monitor.stop()
        if hasattr(_update_display, "live"):
            _update_display.live.stop()
        fail(str(e))
        raise typer.Exit(1)
    finally:
        if hasattr(_update_display, "live"):
            _update_display.live.stop()
        monitor.stop()

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
        ok(f"Dry run complete (no output written)  ({format_eta(monitor.stats.get('elapsed_s', 0))})")
    else:
        ok(f"Saved: {output}  ({format_eta(monitor.stats.get('elapsed_s', 0))})")
