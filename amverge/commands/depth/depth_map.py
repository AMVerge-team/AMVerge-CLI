from __future__ import annotations

from pathlib import Path

import typer

from ...ui import banner, console, err, ok, fail
from ...core.infra.diagnostics import get_gpu_info
from ...core.upscaling.monitor import SystemMonitor, format_eta

def depth_map(
    input: Path = typer.Argument(..., help="Input video file"),
    output: Path = typer.Option(Path("depth_output.mp4"), "--output", "-o", help="Output video file"),
    encoder: str = typer.Option("vits", "--encoder", "-e", help="Model size: vits, vitb, vitl"),
    input_size: int = typer.Option(518, "--input-size", "-s", help="Inference input size (larger = more detail)"),
    pred_only: bool = typer.Option(False, "--pred-only", help="Output depth map only, no original video"),
    grayscale: bool = typer.Option(False, "--grayscale", help="Grayscale depth map (no color palette)"),
    colormap: str = typer.Option("inferno", "--colormap", "-c", help="Color palette: inferno, viridis, plasma, magma, turbo, jet, twilight, hot, cool, rainbow, ocean, bone, winter, summer"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Auto-confirm download prompts"),
    no_monitor: bool = typer.Option(False, "--no-monitor", help="Disable system monitor during processing"),
    ipc: bool = typer.Option(False, "--ipc", hidden=True, help="Emit structured PROGRESS|/PREVIEW_FRAME| events on stderr (app mode)"),
) -> None:
    """Generate depth maps from video using Depth-Anything-V2.

    Per-frame monocular depth estimation. Input video, output side-by-side
    (original + depth) or depth-only video.

    Install: pip install amverge[depth]
    Models auto-downloaded from GitHub Releases on first run.

    Depth-Anything-V2 by Lihe Yang et al. (NeurIPS 2024).
    Based on https://github.com/DepthAnything/Depth-Anything-V2
    Models hosted at https://github.com/AniScripts/AniSmooth-Models
    """
    from ...core.depth import (
        DEPTH_AVAILABLE,
        MODEL_CONFIGS,
        COLMAPS,
        is_model_downloaded,
        download_model,
        generate_depth_map,
    )

    if not DEPTH_AVAILABLE:
        fail(
            "Depth-Anything V2 not installed.\n"
            "  pip install amverge[depth]"
        )
        raise typer.Exit(1)

    if not input.exists():
        fail(f"File not found: {input}")
        raise typer.Exit(1)

    if encoder not in MODEL_CONFIGS:
        fail(f"Unknown encoder: {encoder}. Choose from: {list(MODEL_CONFIGS.keys())}")
        raise typer.Exit(1)

    if colormap not in COLMAPS:
        fail(f"Unknown colormap: {colormap}. Available: {', '.join(COLMAPS.keys())}")
        raise typer.Exit(1)

    if ipc:
        from ...core.infra.ipc import emit_progress
        from ...core.infra.preview import ipc_callbacks

        output.parent.mkdir(parents=True, exist_ok=True)
        progress_cb, preview_cb = ipc_callbacks("depth")
        try:
            generate_depth_map(
                input_path=str(input.resolve()),
                output_path=str(output.resolve()),
                encoder=encoder,
                input_size=input_size,
                pred_only=pred_only,
                grayscale=grayscale,
                colormap=colormap,
                progress_cb=progress_cb,
                preview_cb=preview_cb,
            )
        except Exception as e:
            emit_progress(100, f"Error: {e}")
            raise typer.Exit(1)
        emit_progress(100, f"Saved: {output}")
        return

    banner("depth-map")

    gpu_info = get_gpu_info()
    console.print(f"  Encoder: [accent]{encoder}[/accent]")
    if gpu_info.get("cuda_available"):
        vram = gpu_info.get("vram_gb", 0)
        console.print(f"  GPU: [accent]{gpu_info.get('gpu_name', 'GPU')}[/accent]  "
                      f"VRAM: [accent]{vram:.1f} GB[/accent]")
    console.print(f"  Input size: [accent]{input_size}[/accent]  Colormap: [accent]{colormap}[/accent]")
    if pred_only:
        console.print("  Mode: [accent]depth only[/accent]")
    if grayscale:
        console.print("  Mode: [accent]grayscale[/accent]")
    console.print(f"  Input:  [dim]{input}[/dim]")
    console.print(f"  Output: [dim]{output}[/dim]")

    if not is_model_downloaded(encoder):
        if yes:
            console.print(f"\n  Downloading Depth-Anything-V2-{encoder} model...")
        else:
            if not typer.confirm(
                f"\n  Model 'Depth-Anything-V2-{encoder}' not found. Download now?",
                default=True,
            ):
                fail("Model download cancelled")
                raise typer.Exit(1)

        from ...ui import make_progress as _make_prog
        with _make_prog() as progress:
            task_id = progress.add_task(f"Downloading...", total=100)

            def _download_cb(pct: int, msg: str) -> None:
                progress.update(task_id, completed=pct, description=msg)

            try:
                download_model(encoder, progress_cb=_download_cb)
                ok(f"Model downloaded: Depth-Anything-V2-{encoder}")
            except Exception as e:
                fail(str(e))
                raise typer.Exit(1)

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
            _update_display.task = progress.add_task("Processing depth map...", total=100)
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
        generate_depth_map(
            input_path=str(input.resolve()),
            output_path=str(output.resolve()),
            encoder=encoder,
            input_size=input_size,
            pred_only=pred_only,
            grayscale=grayscale,
            colormap=colormap,
            progress_cb=_progress_cb,
        )
    except ImportError as e:
        monitor.stop()
        if hasattr(_update_display, "live"):
            _update_display.live.stop()
        fail(str(e))
        raise typer.Exit(1)
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

    ok(f"Saved: {output}  ({format_eta(monitor.stats.get('elapsed_s', 0))})")
