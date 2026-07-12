from __future__ import annotations

from pathlib import Path

import typer

from ...ui import banner, console, err, make_progress, ok, fail
from ...core.depth import (
    DEPTH_AVAILABLE,
    MODEL_CONFIGS,
    COLMAPS,
    is_model_downloaded,
    download_model,
    generate_depth_map,
)


def depth_map(
    input: Path = typer.Argument(..., help="Input video file"),
    output: Path = typer.Option(Path("depth_output.mp4"), "--output", "-o", help="Output video file"),
    encoder: str = typer.Option("vits", "--encoder", "-e", help="Model size: vits, vitb, vitl"),
    input_size: int = typer.Option(518, "--input-size", "-s", help="Inference input size (larger = more detail)"),
    pred_only: bool = typer.Option(False, "--pred-only", help="Output depth map only, no original video"),
    grayscale: bool = typer.Option(False, "--grayscale", help="Grayscale depth map (no color palette)"),
    colormap: str = typer.Option("inferno", "--colormap", "-c", help="Color palette: inferno, viridis, plasma, magma, turbo, jet, twilight, hot, cool, rainbow, ocean, bone, winter, summer"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Auto-confirm download prompts"),
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

    banner("depth-map")

    console.print(f"  Encoder: [accent]{encoder}[/accent]")
    console.print(f"  Input size: [accent]{input_size}[/accent]")
    console.print(f"  Colormap: [accent]{colormap}[/accent]")
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

        with make_progress() as progress:
            task_id = progress.add_task(f"Downloading...", total=100)

            def _download_cb(pct: int, msg: str) -> None:
                progress.update(task_id, completed=pct, description=msg)

            try:
                download_model(encoder, progress_cb=_download_cb)
                ok(f"Model downloaded: Depth-Anything-V2-{encoder}")
            except Exception as e:
                fail(str(e))
                raise typer.Exit(1)

    with make_progress() as progress:
        task_id = progress.add_task("Processing depth map...", total=100)

        def _progress_cb(pct: int, msg: str) -> None:
            progress.update(task_id, completed=pct, description=msg)

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
            fail(str(e))
            raise typer.Exit(1)
        except Exception as e:
            fail(str(e))
            raise typer.Exit(1)

    ok(f"Saved: {output}")
