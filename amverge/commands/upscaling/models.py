from __future__ import annotations

import os
from typing import Optional

import typer

from ...ui import banner, console, make_table, ok, fail, dim
from ...core.infra.config import get_models_dir
from ...core.upscaling.registry import (
    UPSCALE_REGISTRY, get_ml_models, get_shader_models, get_onnx_models, get_all_model_keys,
)
from ...core.upscaling.weight_loader import (
    WEIGHTS_DIR, get_weight_path, is_weight_downloaded, download_weights,
)


def _format_size(size_bytes):
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"


def _list_ml_weights():
    entries = []
    for key, entry in get_ml_models().items():
        path = get_weight_path(key)
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        hash_val = entry.get("hash", "")[:12] if entry.get("hash") else "-"
        entries.append((key, entry["file"], _format_size(size) if exists else "-", hash_val, exists))
    return entries


def _list_onnx_models():
    entries = []
    from ...core.upscaling.artcnn import _get_artcnn_dir, ARTCNN_MODELS
    artcnn_dir = _get_artcnn_dir()
    for name, info in ARTCNN_MODELS.items():
        path = os.path.join(artcnn_dir, info["file"])
        exists = os.path.exists(path)
        size = os.path.getsize(path) if exists else 0
        entries.append((name, info["file"], _format_size(size) if exists else "-",
                        info.get("sha256", "-") or "-", exists))
    return entries


def _list_shaders():
    import glob
    from ...core.upscaling.anime4k import _get_anime4k_dir
    shader_dir = _get_anime4k_dir()
    if not os.path.exists(shader_dir):
        return [("Anime4K", "no shaders", "0 B", "not downloaded", False)]
    glsl_files = glob.glob(os.path.join(shader_dir, "*.glsl"))
    total_size = sum(os.path.getsize(f) for f in glsl_files)
    exists = len(glsl_files) > 0
    entry = get_shader_models().get("anime4k", {})
    source = entry.get("credit", "bloc97/Anime4K")
    return [("Anime4K", f"{len(glsl_files)} shaders", _format_size(total_size), source, exists)]


def models(
    delete: Optional[str] = typer.Option(None, "--delete", help="Delete a model by key"),
    download: Optional[str] = typer.Option(None, "--download", help="Download a model by key"),
    show_storage: bool = typer.Option(False, "--storage", help="Show storage location"),
) -> None:
    """Manage upscaling model files.

    Without options, lists all downloaded models and their sizes.
    Use --delete to remove a model, --download to fetch one.
    """
    banner("models")

    if show_storage:
        console.print(f"  Models directory: [dim]{WEIGHTS_DIR}[/dim]")
        from ...core.upscaling.anime4k import _get_anime4k_dir
        from ...core.upscaling.artcnn import _get_artcnn_dir
        console.print(f"  Anime4K shaders:   [dim]{_get_anime4k_dir()}[/dim]")
        console.print(f"  ArtCNN models:     [dim]{_get_artcnn_dir()}[/dim]")
        console.print(f"  FFmpeg:            [dim]{os.path.dirname(WEIGHTS_DIR)}/../../ffmpeg/bin[/dim]")
        return

    if delete:
        all_keys = get_all_model_keys()
        if delete in get_ml_models():
            path = get_weight_path(delete)
            if os.path.exists(path):
                os.unlink(path)
                ok(f"Deleted model: {delete}")
            else:
                fail(f"Model not found on disk: {delete}")
        elif delete == "anime4k":
            import glob
            from ...core.upscaling.anime4k import _get_anime4k_dir
            shader_dir = _get_anime4k_dir()
            deleted = 0
            if os.path.exists(shader_dir):
                for fp in glob.glob(os.path.join(shader_dir, "*.glsl")):
                    os.unlink(fp)
                    deleted += 1
            ok(f"Deleted {deleted} Anime4K shader files")
        elif delete in get_onnx_models():
            from ...core.upscaling.artcnn import _get_artcnn_dir, ARTCNN_MODELS
            info = ARTCNN_MODELS[delete]
            path = os.path.join(_get_artcnn_dir(), info["file"])
            if os.path.exists(path):
                os.unlink(path)
                ok(f"Deleted ArtCNN model: {delete}")
            else:
                fail(f"ArtCNN model not found on disk: {delete}")
        else:
            fail(f"Unknown model key: {delete}. Use --list-models on upscale command.")
            raise typer.Exit(1)
        return

    if download:
        if download in get_ml_models():
            entry = get_ml_models()[download]
            console.print(f"  Downloading [accent]{entry.get('name', download)}[/accent]...")
            success = download_weights(download)
            if success:
                ok(f"Downloaded: {download}")
            else:
                fail(f"Download failed for {download}")
        elif download == "anime4k":
            from ...core.upscaling.anime4k import _download_anime4k_shaders
            console.print("  Downloading [accent]Anime4K shaders[/accent]...")
            shaders = _download_anime4k_shaders()
            ok(f"Downloaded {len(shaders)} Anime4K shader files")
        elif download in get_onnx_models():
            from ...core.upscaling.artcnn import _download_artcnn_model
            entry = get_onnx_models()[download]
            console.print(f"  Downloading [accent]{entry.get('name', download)}[/accent]...")
            _download_artcnn_model(download)
            ok(f"Downloaded ArtCNN model: {download}")
        else:
            fail(f"Unknown model key: {download}")
            raise typer.Exit(1)
        return

    console.print()
    console.print("  [white bold]ML Models[/white bold] [dim](PyTorch / spandrel)[/dim]")
    console.print()
    ml_entries = _list_ml_weights()
    table = make_table(("Model", "bright_black", {}), ("File", "bright_black", {}),
                       ("Size", "bright_black", {}), ("Hash", "bright_black", {}),
                       ("Status", "bright_black", {}), title=None)
    for key, filename, size, hash_short, exists in ml_entries:
        status = "[accent]downloaded[/accent]" if exists else "[muted]not downloaded[/muted]"
        table.add_row(f"[bold]{key}[/bold]", filename, size, f"{hash_short}...", status)
    console.print(table)

    console.print()
    console.print("  [white bold]Shader Upscaler[/white bold]")
    console.print()
    shader_entries = _list_shaders()
    table2 = make_table(("Name", "bright_black", {}), ("Files", "bright_black", {}),
                        ("Size", "bright_black", {}), ("Source", "bright_black", {}),
                        ("Status", "bright_black", {}), title=None)
    for name, files_str, size, source, exists in shader_entries:
        status = "[accent]downloaded[/accent]" if exists else "[muted]not downloaded[/muted]"
        table2.add_row(f"[bold]{name}[/bold]", files_str, size, source, status)
    console.print(table2)

    console.print()
    console.print("  [white bold]ONNX Models[/white bold] [dim](ArtCNN)[/dim]")
    console.print()
    onnx_entries = _list_onnx_models()
    table3 = make_table(("Model", "bright_black", {}), ("File", "bright_black", {}),
                        ("Size", "bright_black", {}), ("Hash", "bright_black", {}),
                        ("Status", "bright_black", {}), title=None)
    for name, filename, size, hash_short, exists in onnx_entries:
        status = "[accent]downloaded[/accent]" if exists else "[muted]not downloaded[/muted]"
        table3.add_row(f"[bold]{name}[/bold]", filename, size, f"{hash_short}", status)
    console.print(table3)

    console.print()
    all_keys = get_all_model_keys()
    console.print(f"  [dim]Use --download <key> to download, --delete <key> to remove[/dim]")
    console.print(f"  [dim]Keys: {', '.join(all_keys)}[/dim]")
