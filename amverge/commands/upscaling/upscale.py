from __future__ import annotations

import os
from pathlib import Path

import typer

from ...ui import banner, console, make_progress, ok, fail, dim
from ...core.infra.diagnostics import get_gpu_info
from ...core.infra.ffmpeg_bootstrap import is_portable_ffmpeg_installed, ensure_ffmpeg
from ...core.upscaling.registry import (
    UPSCALE_REGISTRY,
    QUALITY_PRESETS,
    get_ml_models,
    get_onnx_models,
    get_shader_models,
    get_model_scales,
)


def _ensure_ffmpeg(auto_yes=False):
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


def _ensure_anime4k_shaders(auto_yes=False):
    import glob
    from ...core.upscaling.anime4k import _get_anime4k_dir
    shader_dir = _get_anime4k_dir()
    glsl_files = glob.glob(os.path.join(shader_dir, "*.glsl")) if os.path.exists(shader_dir) else []
    if glsl_files:
        return
    console.print("  [warn]Anime4K shaders not downloaded.[/warn]")
    entry = get_shader_models().get("anime4k", {})
    credit = entry.get("credit", "")
    if auto_yes or typer.confirm(f"  Download Anime4K shaders ({credit})?", default=True):
        from ...core.upscaling.anime4k import _download_anime4k_shaders
        with make_progress() as progress:
            task_id = progress.add_task("Downloading Anime4K shaders...", total=100)
            def _cb(pct, msg):
                progress.update(task_id, completed=pct, description=msg)
            try:
                _download_anime4k_shaders(progress_cb=_cb)
                ok("Anime4K shaders downloaded")
            except Exception as e:
                fail(str(e))
                raise typer.Exit(1)
    else:
        fail("Anime4K shaders are required for this method")
        raise typer.Exit(1)


def _ensure_ml_model(model_key, auto_yes=False):
    from ...core.upscaling.weight_loader import is_weight_downloaded, download_weights
    if is_weight_downloaded(model_key):
        return
    entry = get_ml_models().get(model_key, {})
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


def _ensure_onnx_model(model_name, auto_yes=False):
    from ...core.upscaling.artcnn import _get_artcnn_dir, ARTCNN_MODELS
    info = ARTCNN_MODELS.get(model_name)
    if info is None:
        fail(f"Unknown ONNX model: {model_name}")
        raise typer.Exit(1)
    path = os.path.join(_get_artcnn_dir(), info["file"])
    if os.path.exists(path):
        return
    entry = get_onnx_models().get(model_name, {})
    name = entry.get("name", model_name)
    console.print(f"  [warn]{name} not downloaded.[/warn]")
    if auto_yes or typer.confirm(f"  Download {name}?", default=True):
        from ...core.upscaling.artcnn import _download_artcnn_model
        with make_progress() as progress:
            task_id = progress.add_task(f"Downloading {name}...", total=100)
            def _cb(pct, msg):
                progress.update(task_id, completed=pct, description=msg)
            try:
                _download_artcnn_model(model_name, progress_cb=_cb)
                ok(f"{name} downloaded")
            except Exception as e:
                fail(str(e))
                raise typer.Exit(1)
    else:
        fail(f"{name} is required")
        raise typer.Exit(1)


def upscale(
    input: Path = typer.Argument(None, help="Input video file"),
    output: Path = typer.Option(Path("upscaled.mp4"), "--output", "-o", help="Output video file"),
    method: str = typer.Option("ml", "--method", help="Upscale method: ml, anime4k, artcnn"),
    model: str = typer.Option("adore", "--model", "-m", help="ML model (see --list-models)"),
    onnx_model: str = typer.Option("C4F32", "--onnx-model", help="ONNX model (see --list-models)"),
    anime4k_mode: str = typer.Option("medium", "--anime4k-mode", help="Anime4K mode: light, medium, strong"),
    scale: int = typer.Option(2, "--scale", "-s", help="Scale factor (2 or 4)"),
    preset: str = typer.Option("high", "--preset", "-p", help="Quality: archival, high, balanced, fast, draft"),
    fit_w: int = typer.Option(0, "--fit-w", help="Max output width (0 = no limit)"),
    fit_h: int = typer.Option(0, "--fit-h", help="Max output height (0 = no limit)"),
    list_models: bool = typer.Option(False, "--list-models", help="List available models"),
    credits: bool = typer.Option(False, "--credits", help="Show credits for upscaling technologies"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Auto-confirm all download prompts"),
) -> None:
    """Upscale video using AI super-resolution.

    Methods:
      ml      - Neural network models via PyTorch/spandrel. Based on AniSmooth.
      anime4k - GPU shader-based via FFmpeg filters. Fast, no ML deps.
      artcnn  - Lightweight CNN via ONNX Runtime.

    Use --list-models to see all available models.
    Install: pip install amverge[upscale]
    """
    if list_models:
        banner("upscale models")
        console.print()
        for key, entry in UPSCALE_REGISTRY.items():
            scales_str = "/".join(f"{s}x" for s in entry["scales"])
            console.print(f"  [accent]{key}[/accent] dim:{entry['method']}  "
                          f"{entry['name']} - {scales_str}")
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

    if method not in ("ml", "anime4k", "artcnn"):
        fail(f"Unknown method '{method}'. Valid: ml, anime4k, artcnn")
        raise typer.Exit(1)
    if scale not in (2, 4):
        fail(f"Scale must be 2 or 4, got {scale}")
        raise typer.Exit(1)
    if preset not in QUALITY_PRESETS:
        fail(f"Unknown preset '{preset}'. Valid: {', '.join(QUALITY_PRESETS.keys())}")
        raise typer.Exit(1)

    banner("upscale")

    if method == "anime4k":
        shaders = get_shader_models()
        entry = shaders.get("anime4k", {})
        modes = entry.get("modes", ["medium"])
        if anime4k_mode not in modes:
            fail(f"Unknown Anime4K mode '{anime4k_mode}'. Valid: {', '.join(modes)}")
            raise typer.Exit(1)

        _ensure_ffmpeg(auto_yes=yes)
        _ensure_anime4k_shaders(auto_yes=yes)

        from ...core.upscaling.anime4k import upscale_video_anime4k

        console.print(f"  Method: [accent]{entry.get('name', 'Anime4K')}[/accent] (FFmpeg filters, no ML deps)")
        console.print(f"  Mode: [accent]{anime4k_mode}[/accent]  "
                      f"Scale: [accent]{scale}x[/accent]  "
                      f"Preset: [accent]{preset}[/accent]")
        console.print(f"  Input: [dim]{input}[/dim]")
        console.print(f"  Output: [dim]{output}[/dim]")

        with make_progress() as progress:
            task_id = progress.add_task("Upscaling...", total=100)
            def _progress_cb(pct, msg):
                progress.update(task_id, completed=pct, description=msg)
            try:
                upscale_video_anime4k(
                    str(input.resolve()),
                    str(output.resolve()),
                    scale=scale,
                    mode=anime4k_mode,
                    preset=preset,
                    fit_w=fit_w,
                    fit_h=fit_h,
                    progress_cb=_progress_cb,
                )
            except Exception as e:
                fail(str(e))
                raise typer.Exit(1)

    elif method == "artcnn":
        onnx_models = get_onnx_models()
        if onnx_model not in onnx_models:
            valid = ', '.join(onnx_models.keys())
            fail(f"Unknown ONNX model '{onnx_model}'. Valid: {valid}")
            raise typer.Exit(1)

        _ensure_ffmpeg(auto_yes=yes)
        _ensure_onnx_model(onnx_model, auto_yes=yes)

        from ...core.upscaling.artcnn import upscale_video_artcnn

        entry = onnx_models[onnx_model]
        console.print(f"  Method: [accent]{entry.get('name', onnx_model)}[/accent] (ONNX Runtime)")
        console.print(f"  Model: [accent]{onnx_model}[/accent]  "
                      f"Scale: [accent]{scale}x[/accent]  "
                      f"Preset: [accent]{preset}[/accent]")
        console.print(f"  Input: [dim]{input}[/dim]")
        console.print(f"  Output: [dim]{output}[/dim]")

        with make_progress() as progress:
            task_id = progress.add_task("Upscaling...", total=100)
            def _progress_cb(pct, msg):
                progress.update(task_id, completed=pct, description=msg)
            try:
                upscale_video_artcnn(
                    str(input.resolve()),
                    str(output.resolve()),
                    model_name=onnx_model,
                    scale=scale,
                    preset=preset,
                    fit_w=fit_w,
                    fit_h=fit_h,
                    progress_cb=_progress_cb,
                )
            except Exception as e:
                fail(str(e))
                raise typer.Exit(1)

    else:
        ml_models = get_ml_models()
        if model not in ml_models:
            valid = ', '.join(ml_models.keys())
            fail(f"Unknown ML model '{model}'. Valid: {valid}")
            raise typer.Exit(1)

        model_scales = get_model_scales(model)
        if scale not in model_scales:
            scales_str = '/'.join(f"{s}x" for s in model_scales)
            fail(f"Model '{model}' supports scale {scales_str}, got {scale}x")
            raise typer.Exit(1)

        _ensure_ffmpeg(auto_yes=yes)
        _ensure_ml_model(model, auto_yes=yes)

        entry = ml_models[model]
        try:
            from ...core.upscaling import UPSCALE_AVAILABLE, upscale_video
        except ImportError:
            fail("Upscaling module not available. Dependencies missing.")
            raise typer.Exit(1)

        if not UPSCALE_AVAILABLE:
            fail("ML upscaling requires torch and opencv. Run: pip install amverge[upscale]")
            raise typer.Exit(1)

        gpu_info = get_gpu_info()
        if gpu_info.get("cuda_available"):
            console.print(f"  GPU: [accent]{gpu_info.get('gpu_name', 'N/A')}[/accent]  "
                          f"VRAM: [accent]{gpu_info.get('gpu_memory_free_mb', 0)}/{gpu_info.get('gpu_memory_total_mb', 0)} MB[/accent]")
        else:
            console.print("  [warn]No NVIDIA GPU detected. Upscaling on CPU will be very slow.[/warn]")

        console.print(f"  Model: [accent]{entry.get('name', model)}[/accent]  "
                      f"Scale: [accent]{scale}x[/accent]  "
                      f"Preset: [accent]{preset}[/accent]")
        console.print(f"  Input: [dim]{input}[/dim]")
        console.print(f"  Output: [dim]{output}[/dim]")

        with make_progress() as progress:
            task_id = progress.add_task("Upscaling...", total=100)
            def _progress_cb(pct, msg):
                progress.update(task_id, completed=pct, description=msg)
            try:
                upscale_video(
                    str(input.resolve()),
                    str(output.resolve()),
                    model_name=model,
                    scale=scale,
                    preset=preset,
                    fit_w=fit_w,
                    fit_h=fit_h,
                    progress_cb=_progress_cb,
                )
            except Exception as e:
                fail(str(e))
                raise typer.Exit(1)

    ok(f"Upscaled video saved to {output}")
