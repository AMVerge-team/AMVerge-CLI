from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import typer

from ...ui import banner, console, err, make_progress, ok, fail, dim
from ...core.infra.diagnostics import get_gpu_info
from ...core.infra.ffmpeg_bootstrap import is_portable_ffmpeg_installed, ensure_ffmpeg
from ...core.upscaling.registry import (
    UPSCALE_REGISTRY,
    QUALITY_PRESETS,
    get_model_scales,
)


def _sample_gpu():
    try:
        import subprocess
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
            creationflags=0x08000000 if os.name == "nt" else 0,
        )
        if smi.returncode == 0 and smi.stdout.strip():
            parts = [p.strip() for p in smi.stdout.strip().split(",")]
            if len(parts) >= 4:
                return {
                    "gpu_util": float(parts[0]),
                    "vram_used": float(parts[1]),
                    "vram_total": float(parts[2]),
                    "gpu_temp": float(parts[3]),
                }
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            return {
                "gpu_util": None,
                "vram_used": (total - free) / (1024 * 1024),
                "vram_total": total / (1024 * 1024),
                "gpu_temp": None,
            }
    except Exception:
        pass

    return None


def _sample_cpu():
    try:
        import psutil
        return {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_used": psutil.virtual_memory().used / (1024 ** 3),
            "ram_total": psutil.virtual_memory().total / (1024 ** 3),
        }
    except ImportError:
        return None


def _format_eta(seconds):
    if seconds is None or seconds == float("inf"):
        return "--:--"
    seconds = int(seconds)
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


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
    from ...core.upscaling.weight_loader import is_weight_downloaded, download_weights
    if is_weight_downloaded(model_key):
        return
    entry = UPSCALE_REGISTRY.get(model_key, {})
    name = entry.get("name", model_key)
    console.print(f"  [warn]Model '{name}' not downloaded.[/warn]")
    if auto_yes or typer.confirm(f"  Download {name}?", default=True):
        with make_progress() as progress:
            task_id = progress.add_task(f"Downloading {name}...", total=100)
            def _cb(pct, msg):
                progress.update(task_id, completed=pct, description=msg)
            if not download_weights(model_key, progress_cb=_cb):
                fail(f"Download failed for {name}")
                raise typer.Exit(1)
            ok(f"Model {name} downloaded")
    else:
        fail(f"Model {name} is required")
        raise typer.Exit(1)


def upscale(
    model: str = typer.Option("adore", "--model", "-m", help="Model key from registry (see --list-models)"),
    input: Path = typer.Argument(None, help="Input video file"),
    output: Path = typer.Option(Path("upscaled.mp4"), "--output", "-o", help="Output video file"),
    scale: int = typer.Option(2, "--scale", "-s", help="Scale factor (model-specific)"),
    preset: str = typer.Option("high", "--preset", "-p", help="Quality: archival, high, balanced, fast, draft"),
    mode: str = typer.Option("medium", "--mode", help="Shader mode: light, medium, strong"),
    fit_w: int = typer.Option(0, "--fit-w", help="Max output width (0 = no limit)"),
    fit_h: int = typer.Option(0, "--fit-h", help="Max output height (0 = no limit)"),
    list_models: bool = typer.Option(False, "--list-models", help="List all available models"),
    credits: bool = typer.Option(False, "--credits", help="Show credits for upscaling technologies"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Auto-confirm all download prompts"),
    no_monitor: bool = typer.Option(False, "--no-monitor", help="Disable system monitor during upscale"),
) -> None:
    """Upscale video using AI super-resolution.

    All models and methods are defined in registry.json. Use --list-models to browse.
    """
    if list_models:
        banner("upscale models")
        console.print()
        for key, entry in UPSCALE_REGISTRY.items():
            scales_str = "/".join(f"{s}x" for s in entry["scales"])
            console.print(f"  [accent]{key}[/accent]  {entry['name']}  dim:{entry['method']}  {scales_str}")
            console.print(f"    {entry.get('description', '')}")
            console.print(f"    Credit: {entry.get('credit', '')}")
        console.print()
        return

    if credits:
        banner("upscale credits")
        console.print()
        seen = set()
        for entry in UPSCALE_REGISTRY.values():
            cred = entry.get("credit", "")
            if cred and cred not in seen:
                console.print(f"  [accent]+[/accent] {cred}")
                seen.add(cred)
        console.print()
        return

    if input is None:
        fail("Missing INPUT argument.")
        raise typer.Exit(1)
    if not input.exists():
        fail(f"File not found: {input}")
        raise typer.Exit(1)

    if model not in UPSCALE_REGISTRY:
        fail(f"Unknown model '{model}'. Use --list-models to see available models.")
        raise typer.Exit(1)

    entry = UPSCALE_REGISTRY[model]
    method = entry["method"]

    valid_scales = get_model_scales(model)
    if scale not in valid_scales:
        scales_str = "/".join(f"{s}x" for s in valid_scales)
        fail(f"Model '{model}' supports {scales_str}, got {scale}x")
        raise typer.Exit(1)

    if preset not in QUALITY_PRESETS:
        fail(f"Unknown preset '{preset}'. Valid: {', '.join(QUALITY_PRESETS.keys())}")
        raise typer.Exit(1)

    banner("upscale")

    if method in ("ml", "onnx"):
        try:
            from ...core.upscaling.engine import UPSCALE_AVAILABLE
        except ImportError:
            fail("Upscaling module not available. Dependencies missing.")
            raise typer.Exit(1)

        if method == "ml" and not UPSCALE_AVAILABLE:
            fail("ML upscaling requires torch and opencv. Run: pip install amverge[upscale]")
            raise typer.Exit(1)

        _ensure_ffmpeg_interactive(auto_yes=yes)
        _ensure_model_downloaded(model, auto_yes=yes)

        if method == "ml":
            gpu_info = get_gpu_info()
            if gpu_info.get("cuda_available"):
                vram = gpu_info.get("vram_gb", 0)
                console.print(f"  GPU: [accent]{gpu_info.get('gpu_name', 'N/A')}[/accent]  "
                              f"VRAM: [accent]{vram:.1f} GB[/accent]")
            else:
                console.print("  [warn]No NVIDIA GPU detected. Upscaling on CPU will be very slow.[/warn]")

        console.print(f"  Model: [accent]{entry.get('name', model)}[/accent]  "
                      f"Scale: [accent]{scale}x[/accent]  "
                      f"Preset: [accent]{preset}[/accent]")
        console.print(f"  Input: [dim]{input}[/dim]")
        console.print(f"  Output: [dim]{output}[/dim]")

    elif method == "shader":
        _ensure_ffmpeg_interactive(auto_yes=yes)

        shader_modes = entry.get("modes", ["medium"])
        if mode not in shader_modes:
            fail(f"Unknown mode '{mode}'. Valid: {', '.join(shader_modes)}")
            raise typer.Exit(1)

        console.print(f"  Method: [accent]{entry.get('name', model)}[/accent] (FFmpeg filters, no ML)")
        console.print(f"  Mode: [accent]{mode}[/accent]  "
                      f"Scale: [accent]{scale}x[/accent]  "
                      f"Preset: [accent]{preset}[/accent]")
        console.print(f"  Input: [dim]{input}[/dim]")
        console.print(f"  Output: [dim]{output}[/dim]")

    from ...core.upscaling.engine import upscale_model

    monitor_data = {"pct": 0, "msg": "Starting...", "running": True, "done": False}
    monitor_thread = None
    start_time = time.time()

    if not no_monitor and method in ("ml", "onnx"):
        monitor_data["gpu_name"] = get_gpu_info().get("gpu_name", "GPU")

        def _monitor_loop():
            while monitor_data["running"]:
                gpu = _sample_gpu()
                cpu = _sample_cpu()
                if gpu:
                    monitor_data.update(gpu)
                if cpu:
                    monitor_data.update(cpu)
                time.sleep(1.0)

        monitor_thread = threading.Thread(target=_monitor_loop, daemon=True)
        monitor_thread.start()

    last_pct = 0

    def _progress_cb(pct, msg):
        nonlocal last_pct
        last_pct = pct
        now = time.time()
        monitor_data["pct"] = pct
        monitor_data["msg"] = msg
        elapsed = now - start_time
        if pct > 0:
            eta = (elapsed / pct) * (100 - pct) if pct < 100 else 0
            monitor_data["eta"] = eta
            monitor_data["fps"] = (60 * 2.5 * pct / 100) / elapsed if elapsed > 0 else 0
        monitor_data["elapsed"] = elapsed
        _update_display(monitor_data, no_monitor)

    def _update_display(data, skip_monitor):
        from rich.live import Live
        from rich.panel import Panel
        from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn

        if not hasattr(_update_display, "live"):
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
            )
            _update_display.task = progress.add_task("Upscaling...", total=100)
            _update_display.progress = progress
            _update_display.live = Live(progress, console=err, refresh_per_second=4, transient=True)
            _update_display.live.start()

        _update_display.progress.update(_update_display.task, completed=data["pct"], description=data["msg"])

        if not skip_monitor and hasattr(_update_display, "live"):
            lines = []
            gpu_name = data.get("gpu_name", "GPU")
            gpu_parts = []
            if data.get("gpu_util") is not None:
                gpu_parts.append(f"GPU {data['gpu_util']:.0f}%")
            if data.get("gpu_temp") is not None:
                gpu_parts.append(f"{data['gpu_temp']:.0f}°C")
            if data.get("vram_used") is not None and data.get("vram_total"):
                gpu_parts.append(f"VRAM {data['vram_used']:.0f}/{data['vram_total']:.0f} MB")
            if gpu_parts:
                lines.append(f"  {gpu_name}: {' | '.join(gpu_parts)}")

            cpu_parts = []
            if data.get("cpu_percent") is not None:
                cpu_parts.append(f"CPU {data['cpu_percent']:.0f}%")
            if data.get("ram_used") is not None and data.get("ram_total"):
                cpu_parts.append(f"RAM {data['ram_used']:.1f}/{data['ram_total']:.1f} GB")
            if cpu_parts:
                lines.append(f"  {' | '.join(cpu_parts)}")

            status_parts = []
            if data.get("eta") is not None and data["eta"] != float("inf"):
                status_parts.append(f"ETA {_format_eta(data['eta'])}")
            if data.get("elapsed"):
                status_parts.append(f"elapsed {_format_eta(data['elapsed'])}")
            if data.get("fps") and data["fps"] > 0:
                status_parts.append(f"~{data['fps']:.1f} fps")
            if status_parts:
                lines.append(f"  {' | '.join(status_parts)}")

            from rich.console import Group
            content = [_update_display.progress]
            if lines:
                content.append(Panel("\n".join(lines), border_style="#22c55e", padding=(0, 1)))
            _update_display.live.update(Group(*content))

    try:
        upscale_model(
            model_key=model,
            input_path=str(input.resolve()),
            output_path=str(output.resolve()),
            scale=scale,
            preset=preset,
            fit_w=fit_w,
            fit_h=fit_h,
            mode=mode if method == "shader" else None,
            progress_cb=_progress_cb,
        )
    except Exception as e:
        monitor_data["running"] = False
        if hasattr(_update_display, "live"):
            _update_display.live.stop()
        fail(str(e))
        raise typer.Exit(1)
    finally:
        if hasattr(_update_display, "live"):
            _update_display.live.stop()

    elapsed = time.time() - start_time
    monitor_data["done"] = True
    monitor_data["running"] = False
    ok(f"Upscaled video saved to {output} ({elapsed:.1f}s)")
