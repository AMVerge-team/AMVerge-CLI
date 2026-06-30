from __future__ import annotations

from pathlib import Path

import typer

from ...ui import banner, console, err, make_progress, ok, fail
from ...core.infra.diagnostics import get_gpu_info
from ...core.interpolation.registry import (
    INTERPOLATION_REGISTRY,
    QUALITY_PRESETS,
)


def interpolate(
    input: Path = typer.Argument(..., help="Input video file"),
    output: Path = typer.Option(Path("interpolated.mp4"), "--output", "-o", help="Output video file"),
    model: str = typer.Option("rife4.25", "--model", "-m", help="Interpolation model key from registry"),
    factor: int = typer.Option(2, "--factor", "-f", help="Frame rate multiplier (2-16)"),
    preset: str = typer.Option("high", "--preset", "-p", help="Quality: archival, high, balanced, fast, draft"),
    list_models: bool = typer.Option(False, "--list-models", help="List all available models"),
    credits: bool = typer.Option(False, "--credits", help="Show credits for interpolation technologies"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Auto-confirm download prompts"),
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

    if not input.exists():
        fail(f"File not found: {input}")
        raise typer.Exit(1)

    if model not in INTERPOLATION_REGISTRY:
        fail(f"Unknown model '{model}'. Use --list-models to see available models.")
        raise typer.Exit(1)

    if factor < 2 or factor > 16:
        fail("Factor must be between 2 and 16")
        raise typer.Exit(1)

    if preset not in QUALITY_PRESETS:
        fail(f"Unknown preset '{preset}'. Valid: {', '.join(QUALITY_PRESETS.keys())}")
        raise typer.Exit(1)

    from ...core.interpolation.engine import INTERPOLATION_AVAILABLE
    if not INTERPOLATION_AVAILABLE:
        fail("Interpolation requires torch and opencv. Run: pip install amverge[ml]")
        raise typer.Exit(1)

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

    from ...core.interpolation import is_weight_downloaded, download_weights

    if not is_weight_downloaded(model) and not yes:
        console.print(f"\n  [warn]Model '{entry['name']}' not downloaded.[/warn]")
        typer.confirm(f"  Download {entry['name']}?", default=True, abort=True)

    if not is_weight_downloaded(model):
        with make_progress() as progress:
            task_id = progress.add_task(f"Downloading {entry['name']}...", total=100)
            def _dl_cb(pct, msg):
                progress.update(task_id, completed=pct, description=msg)
            try:
                download_weights(model, progress_cb=_dl_cb)
            except Exception as e:
                fail(f"Download failed: {e}")
                raise typer.Exit(1)

    from ...core.interpolation import interpolate_video

    with make_progress() as progress:
        task_id = progress.add_task("Interpolating...", total=100)

        def _progress_cb(pct, msg):
            progress.update(task_id, completed=pct, description=msg)

        try:
            interpolate_video(
                input_path=str(input.resolve()),
                output_path=str(output.resolve()),
                model_key=model,
                factor=factor,
                preset=preset,
                progress_cb=_progress_cb,
            )
        except Exception as e:
            fail(str(e))
            raise typer.Exit(1)

    ok(f"Saved: {output}")
