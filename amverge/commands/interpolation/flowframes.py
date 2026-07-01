from __future__ import annotations

import os
from pathlib import Path

import typer

from ...ui import banner, console, err, make_progress, ok, fail
from ...core.infra.diagnostics import get_gpu_info
from ...core.infra.ffmpeg_bootstrap import is_portable_ffmpeg_installed, ensure_ffmpeg
from ...core.upscaling.monitor import SystemMonitor, format_eta
from ...core.interpolation import (
    flowframes_available,
    run_flowframes,
    get_flowframes_path,
    set_flowframes_path,
)


def _ensure_ffmpeg_interactive(auto_yes=False):
    if not is_portable_ffmpeg_installed():
        console.print("  [warn]FFmpeg not found on your system.[/warn]")
        if auto_yes or typer.confirm("  Download portable FFmpeg?", default=True):
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


def flowframes(
    input: Path = typer.Argument(..., help="Input video file"),
    output: Path = typer.Option(Path("interpolated.mp4"), "--output", "-o", help="Output video file"),
    factor: int = typer.Option(2, "--factor", "-f", help="Frame rate multiplier (2x-16x)"),
    ai: str = typer.Option("RifeNcnn", "--ai", help="AI implementation for Flowframes 1.42.0"),
    model: str = typer.Option("RIFE 4.26", "--model", "-m", help="Interpolation model name"),
    output_format: str = typer.Option("Mp4", "--format", help="Output format"),
    encoder: str = typer.Option("X264", "--encoder", "-e", help="Video encoder"),
    pix_fmt: str = typer.Option("Yuv420P", "--pix-fmt", help="Pixel format"),
    quality: float = typer.Option(None, "--quality", "-q", help="Quality setting (integer, higher = better)"),
    max_fps: float = typer.Option(None, "--max-fps", help="Max output FPS cap"),
    max_height: int = typer.Option(None, "--max-height", help="Max output height"),
    scene_change: bool = typer.Option(False, "--scene-change", help="Enable scene change detection"),
    scene_sensitivity: float = typer.Option(None, "--scene-sensitivity", help="Scene change sensitivity"),
    ff_path: Path = typer.Option(None, "--ff-path", help="Path to Flowframes.exe"),
    timeout: float = typer.Option(36000.0, "--timeout", help="Max runtime in seconds (default 10 hours)"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Auto-confirm download prompts"),
    no_monitor: bool = typer.Option(False, "--no-monitor", help="Disable system monitor during interpolation"),
) -> None:
    """Run Flowframes 1.42.0 frame interpolation.

    Requires Flowframes 1.42.0 Patreon installed. Auto-detects at %LOCALAPPDATA%\\Flowframes\\Flowframes.exe
    or configure with: amverge flowframes-path PATH

    Support for free Flowframes 1.36.0 is planned (delivery TBD - differs from 1.42.0 Patreon version).
    """
    if ff_path:
        if not ff_path.exists():
            fail(f"Flowframes.exe not found: {ff_path}")
            raise typer.Exit(1)
        set_flowframes_path(str(ff_path))
        ok(f"Flowframes path saved: {ff_path}")

    if not input.exists():
        fail(f"File not found: {input}")
        raise typer.Exit(1)

    if not flowframes_available():
        fail(
            "Flowframes.exe (1.42.0) not found.\n"
            "  Set path:  amverge flowframes-path PATH\n"
            "  Free Flowframes 1.36.0 support is planned (delivery TBD)."
        )
        raise typer.Exit(1)

    if factor < 2 or factor > 16:
        fail("Factor must be between 2 and 16")
        raise typer.Exit(1)

    _ensure_ffmpeg_interactive(auto_yes=yes)

    banner("flowframes")

    ff_exe = get_flowframes_path()
    gpu_info = get_gpu_info()
    console.print(f"  Flowframes: [accent]{ff_exe}[/accent]")
    if gpu_info.get("cuda_available"):
        vram = gpu_info.get("vram_gb", 0)
        console.print(f"  GPU: [accent]{gpu_info.get('gpu_name', 'N/A')}[/accent]  "
                      f"VRAM: [accent]{vram:.1f} GB[/accent]")
    console.print(f"  AI: [accent]{ai}[/accent]  Model: [accent]{model}[/accent]  Factor: [accent]{factor}x[/accent]")
    console.print(f"  Input:  [dim]{input}[/dim]")
    console.print(f"  Output: [dim]{output}[/dim]")

    output_dir = str(output.parent.resolve()) if output.parent != Path(".") else os.path.join(os.getcwd(), ".")
    output_basename = output.name

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
            _update_display.task = progress.add_task("Interpolating (Flowframes)...", total=100)
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

    def _progress_cb(pct: int, msg: str) -> None:
        monitor.progress_callback(pct, msg)
        _update_display()

    try:
        produced = run_flowframes(
            input_path=str(input.resolve()),
            output_dir=output_dir,
            factor=factor,
            ai=ai,
            model=model,
            output_format=output_format,
            encoder=encoder,
            pix_fmt=pix_fmt,
            quality=int(quality) if quality is not None else None,
            max_fps=max_fps,
            max_height=max_height,
            scene_change=scene_change,
            scene_sensitivity=scene_sensitivity,
            progress_cb=_progress_cb,
            timeout=timeout,
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

    produced_path = Path(produced)
    expected_path = Path(output_dir) / output_basename

    if produced_path != expected_path:
        try:
            produced_path.rename(expected_path)
            ok(f"Renamed output: {expected_path} ({monitor.stats['elapsed_s']:.1f}s)")
        except OSError:
            ok(f"Saved: {produced_path} ({monitor.stats['elapsed_s']:.1f}s)")
    else:
        ok(f"Saved: {expected_path} ({monitor.stats['elapsed_s']:.1f}s)")
