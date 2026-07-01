from __future__ import annotations

from pathlib import Path

import typer

from ...ui import banner, console, err, make_progress, ok, fail
from ...core.infra.diagnostics import get_gpu_info
from ...core.upscaling.monitor import SystemMonitor, format_eta
from ...core.dedup import DEDUP_METHODS


def dedup(
    input: Path = typer.Argument(..., help="Input video file"),
    output: Path = typer.Option(None, "--output", "-o", help="Output video file"),
    method: str = typer.Option("advanced", "--method", "-m", help="Dedup method: ffmpeg, ssim, framediff, advanced"),
    threshold: float = typer.Option(None, "--threshold", "-t", help="Detection threshold (method-specific)"),
    min_change_pct: float = typer.Option(2.0, "--min-change-pct", help="Min changed pixel %% for framediff method"),
    region_sensitivity: int = typer.Option(1, "--region-sensitivity", "-rs", help="Min regions to change for advanced (1-4)"),
    no_optical_flow: bool = typer.Option(False, "--no-optical-flow", help="Disable optical flow in advanced method"),
    no_camera_comp: bool = typer.Option(False, "--no-camera-comp", help="Disable camera motion compensation in advanced"),
    keep_camera_only: bool = typer.Option(False, "--keep-camera-only", help="Keep static-subject frames in advanced"),
    list_methods: bool = typer.Option(False, "--list-methods", help="List available dedup methods"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Auto-confirm download prompts"),
    no_monitor: bool = typer.Option(False, "--no-monitor", help="Disable system monitor during dedup"),
    gpu: bool = typer.Option(False, "--gpu", help="Enable GPU acceleration for advanced method (needs OpenCV CUDA)"),
) -> None:
    """Remove duplicate / dead frames from a video.

    Supports four methods: ffmpeg (mpdecimate), ssim (OpenCV), framediff (OpenCV),
    and advanced (optical flow + camera compensation + static subject detection).
    Advanced is the default and best for anime content.
    """
    if list_methods:
        banner("dedup methods")
        console.print()
        for key, entry in DEDUP_METHODS.items():
            req = entry.get("requires") or "none"
            default_t = {"ffmpeg": 2.0, "ssim": 0.987, "framediff": 10.0, "advanced": 0.95}.get(key, "-")
            tag = " [default]" if key == "advanced" else ""
            console.print(f"  [accent]{key}[/accent]{tag} - {entry['name']}")
            console.print(f"    {entry['description']}")
            console.print(f"    Default threshold: [dim]{default_t}[/]")
            console.print(f"    Requires: [dim]{req}[/]")
        console.print()
        return

    if not input.exists():
        fail(f"File not found: {input}")
        raise typer.Exit(1)

    if method not in DEDUP_METHODS:
        fail(f"Unknown method '{method}'. Valid: {', '.join(DEDUP_METHODS.keys())}")
        raise typer.Exit(1)

    if threshold is None:
        threshold = {"ffmpeg": 2.0, "ssim": 0.987, "framediff": 10.0, "advanced": 0.95}.get(method, 2.0)

    if output is None:
        output = input.parent / f"{input.stem}_deduped{input.suffix}"

    from ...core.infra.ffmpeg_bootstrap import is_portable_ffmpeg_installed, ensure_ffmpeg

    def _ensure_ff():
        if not is_portable_ffmpeg_installed():
            console.print("  [warn]FFmpeg not found on your system.[/warn]")
            if yes or typer.confirm("  Download portable FFmpeg?", default=True):
                with make_progress() as progress:
                    task_id = progress.add_task("Downloading FFmpeg...", total=100)
                    def _cb(pct, msg):
                        progress.update(task_id, completed=pct, description=msg)
                    try:
                        ensure_ffmpeg(progress_cb=_cb)
                        ok("FFmpeg installed")
                    except Exception as e:
                        fail(str(e))
                        raise typer.Exit(1)
            else:
                fail("FFmpeg is required: https://ffmpeg.org/download.html")
                raise typer.Exit(1)

    _ensure_ff()

    entry = DEDUP_METHODS[method]

    banner("dedup")

    gpu_info = get_gpu_info()
    console.print(f"  Method: [accent]{entry['name']}[/accent]")
    console.print(f"  Threshold: [accent]{threshold}[/accent]")
    if method == "framediff":
        console.print(f"  Min change: [accent]{min_change_pct}%[/accent]")
    if method == "advanced":
        console.print(f"  Region sensitivity: [accent]{region_sensitivity}[/accent]")
        console.print(f"  Optical flow: [accent]{'off' if no_optical_flow else 'on'}[/accent]")
        console.print(f"  Camera comp: [accent]{'off' if no_camera_comp else 'on'}[/accent]")
        console.print(f"  Static subject: [accent]{'keep' if keep_camera_only else 'remove'}[/accent]")
        console.print(f"  GPU: [accent]{'on' if gpu else 'off'}[/accent]")
    console.print(f"  Input:  [dim]{input}[/dim]")
    console.print(f"  Output: [dim]{output}[/dim]")

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
                gpu_parts.append(f"{s['gpu_temp']:.0f}C")
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
        if method == "ffmpeg":
            from ...core.dedup import dedup_ffmpeg
            _, stats = dedup_ffmpeg(str(input.resolve()), str(output.resolve()), threshold, _progress_cb)
        elif method == "ssim":
            from ...core.dedup import dedup_ssim, SSIM_AVAILABLE
            if not SSIM_AVAILABLE:
                monitor.stop()
                if hasattr(_update_display, "live"):
                    _update_display.live.stop()
                fail("SSIM method requires opencv and scikit-image. Run: pip install opencv-python scikit-image")
                raise typer.Exit(1)
            _, stats = dedup_ssim(str(input.resolve()), str(output.resolve()), threshold, _progress_cb)
        elif method == "framediff":
            from ...core.dedup import dedup_framediff, FRAMEDIFF_AVAILABLE
            if not FRAMEDIFF_AVAILABLE:
                monitor.stop()
                if hasattr(_update_display, "live"):
                    _update_display.live.stop()
                fail("FrameDiff method requires opencv. Run: pip install opencv-python")
                raise typer.Exit(1)
            _, stats = dedup_framediff(str(input.resolve()), str(output.resolve()), threshold, min_change_pct, _progress_cb)
        elif method == "advanced":
            from ...core.dedup import dedup_advanced, ADVANCED_AVAILABLE
            if not ADVANCED_AVAILABLE:
                monitor.stop()
                if hasattr(_update_display, "live"):
                    _update_display.live.stop()
                fail("Advanced method requires opencv. Run: pip install opencv-python")
                raise typer.Exit(1)
            _, stats = dedup_advanced(
                str(input.resolve()), str(output.resolve()), threshold,
                region_sensitivity=region_sensitivity,
                use_optical_flow=not no_optical_flow,
                camera_motion_compensation=not no_camera_comp,
                remove_static_subject=not keep_camera_only,
                use_gpu=gpu,
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
        console.print()
        console.print(f"  Frames in:    [label]{stats['frames_in']}[/]")
        console.print(f"  Frames kept:  [accent]{stats['frames_out']}[/]")
        console.print(f"  Removed:      [warn]{stats['frames_removed']}[/]  ([warn]{stats['pct_removed']}%[/])")
        if stats.get("cadence"):
            console.print(f"  Cadence:      [dim]every {stats['cadence']} frames[/]")
        ok(f"Saved: {output} ({monitor.stats['elapsed_s']:.1f}s)")
