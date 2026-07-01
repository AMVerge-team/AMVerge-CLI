from __future__ import annotations

from pathlib import Path

import typer

from ...ui import banner, console, err, make_progress, ok, fail
from ...core.infra.diagnostics import get_gpu_info
from ...core.infra.ffmpeg_bootstrap import is_portable_ffmpeg_installed, ensure_ffmpeg
from ...core.upscaling.monitor import SystemMonitor, format_eta
from ...core.interpolation.registry import (
    INTERPOLATION_REGISTRY,
    QUALITY_PRESETS,
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


def _ensure_model_downloaded(model_key, auto_yes=False):
    from ...core.interpolation import is_weight_downloaded, download_weights

    if is_weight_downloaded(model_key):
        return

    entry = INTERPOLATION_REGISTRY.get(model_key, {})
    name = entry.get("name", model_key)
    console.print(f"  [warn]Model '{name}' not downloaded.[/warn]")
    if auto_yes or typer.confirm(f"  Download {name}?", default=True):
        with make_progress() as progress:
            task_id = progress.add_task(f"Downloading {name}...", total=100)
            def _cb(pct, msg):
                progress.update(task_id, completed=pct, description=msg)
            try:
                download_weights(model_key, progress_cb=_cb)
            except Exception as e:
                fail(f"Download failed for {name}: {e}")
                raise typer.Exit(1)
            ok(f"Model {name} downloaded")
    else:
        fail(f"Model {name} is required")
        raise typer.Exit(1)


def interpolate(
    input: Path = typer.Argument(None, help="Input video file"),
    output: Path = typer.Option(Path("interpolated.mp4"), "--output", "-o", help="Output video file"),
    model: str = typer.Option("rife4.25", "--model", "-m", help="Interpolation model key from registry"),
    factor: int = typer.Option(2, "--factor", "-f", help="Frame rate multiplier (2-64)"),
    preset: str = typer.Option("high", "--preset", "-p", help="Quality: archival, high, balanced, fast, draft"),
    target_size_mb: float = typer.Option(0, "--target-size-mb", help="Target output file size in MB (two-pass encode)"),
    fit_w: int = typer.Option(0, "--fit-w", help="Max output width (0 = no limit)"),
    fit_h: int = typer.Option(0, "--fit-h", help="Max output height (0 = no limit)"),
    list_models: bool = typer.Option(False, "--list-models", help="List all available models"),
    credits: bool = typer.Option(False, "--credits", help="Show credits for interpolation technologies"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Auto-confirm download prompts"),
    download: bool = typer.Option(False, "--download", help="Download model weights without running"),
    no_monitor: bool = typer.Option(False, "--no-monitor", help="Disable system monitor during interpolation"),
) -> None:
    """Interpolate video frames using AI frame interpolation (RIFE).

    Python-based RIFE inference with PyTorch CUDA/CPU. Requires PyTorch and OpenCV.
    For Flowframes 1.42.0 external process, use: amverge flowframes
    """
    if list_models:
        banner("interpolate models")
        console.print()
        for key, entry in INTERPOLATION_REGISTRY.items():
            console.print(f"  [accent]{key}[/accent]  {entry['name']}  dim:{entry['method']}")
            console.print(f"    {entry.get('description', '')}")
            console.print(f"    Credit: {entry.get('credit', '')}")
        console.print()
        return

    if credits:
        banner("interpolate credits")
        console.print()
        seen = set()
        for entry in INTERPOLATION_REGISTRY.values():
            cred = entry.get("credit", "")
            if cred and cred not in seen:
                console.print(f"  [accent]+[/accent] {cred}")
                seen.add(cred)
        console.print()
        return

    from ...core.interpolation import download_weights as _interp_dl

    if download:
        if model not in INTERPOLATION_REGISTRY:
            fail(f"Unknown model '{model}'. Use --list-models to see available models.")
            raise typer.Exit(1)
        entry = INTERPOLATION_REGISTRY[model]
        console.print(f"  Downloading [accent]{entry['name']}[/accent]...")
        with make_progress() as progress:
            task_id = progress.add_task(f"Downloading {entry['name']}...", total=100)
            def _dl_cb(pct, msg):
                progress.update(task_id, completed=pct, description=msg)
            _interp_dl(model, progress_cb=_dl_cb)
        ok(f"Downloaded: {model}")
        return

    if input is None:
        fail("Missing INPUT argument.")
        raise typer.Exit(1)
    if not input.exists():
        fail(f"File not found: {input}")
        raise typer.Exit(1)

    if model not in INTERPOLATION_REGISTRY:
        fail(f"Unknown model '{model}'. Use --list-models to see available models.")
        raise typer.Exit(1)

    if factor < 2 or factor > 64:
        fail("Factor must be between 2 and 64")
        raise typer.Exit(1)

    if preset not in QUALITY_PRESETS:
        fail(f"Unknown preset '{preset}'. Valid: {', '.join(QUALITY_PRESETS.keys())}")
        raise typer.Exit(1)

    from ...core.interpolation.engine import INTERPOLATION_AVAILABLE
    if not INTERPOLATION_AVAILABLE:
        fail("Interpolation requires torch and opencv. Run: pip install amverge[ml]")
        raise typer.Exit(1)

    _ensure_ffmpeg_interactive(auto_yes=yes)
    _ensure_model_downloaded(model, auto_yes=yes)

    banner("interpolate")

    gpu_info = get_gpu_info()
    if gpu_info.get("cuda_available"):
        vram = gpu_info.get("vram_gb", 0)
        console.print(f"  GPU: [accent]{gpu_info.get('gpu_name', 'N/A')}[/accent]  "
                      f"VRAM: [accent]{vram:.1f} GB[/accent]")
    else:
        console.print("  [warn]No NVIDIA GPU detected. Interpolation on CPU will be very slow.[/warn]")

    entry = INTERPOLATION_REGISTRY[model]
    console.print(f"  Model: [accent]{entry['name']}[/accent]  "
                  f"Factor: [accent]{factor}x[/accent]  "
                  f"Preset: [accent]{preset}[/accent]")
    console.print(f"  Input:  [dim]{input}[/dim]")
    console.print(f"  Output: [dim]{output}[/dim]")
    if target_size_mb > 0:
        console.print(f"  Target: [accent]{target_size_mb:.0f} MB[/accent]")
    if fit_w > 0 or fit_h > 0:
        console.print(f"  Fit:    [accent]{fit_w}x{fit_h}[/accent]")

    from ...core.interpolation import interpolate_video

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
            _update_display.task = progress.add_task("Interpolating...", total=100)
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

    try:
        interpolate_video(
            input_path=str(input.resolve()),
            output_path=str(output.resolve()),
            model_key=model,
            factor=factor,
            preset=preset,
            target_size_mb=target_size_mb,
            fit_w=fit_w,
            fit_h=fit_h,
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
    ok(f"Saved: {output} ({monitor.stats['elapsed_s']:.1f}s)")
