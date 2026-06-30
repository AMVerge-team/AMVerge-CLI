from __future__ import annotations

import os
from typing import Optional

import typer

from ...ui import banner, console, make_table, ok, fail
from ...core.infra.config import get_models_dir
from ...core.upscaling.registry import (
    UPSCALE_REGISTRY, get_ml_models, get_onnx_models,
)
from ...core.upscaling.weight_loader import (
    WEIGHTS_DIR, get_weight_path, is_weight_downloaded, download_weights,
)
from ...core.upscaling.anime4k import (
    download_anime4k_shaders, is_anime4k_downloaded, list_shaders, get_shader_dir,
)
from ...core.upscaling.artcnn import (
    download_artcnn, is_artcnn_downloaded, get_artcnn_path,
)


def _format_size(size_bytes):
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"


def _get_model_size(key):
    entry = UPSCALE_REGISTRY.get(key, {})

    if entry.get("method") == "ml":
        path = get_weight_path(key)
    elif entry.get("method") == "onnx":
        path = get_artcnn_path(key)
    elif entry.get("method") == "shader":
        shader_dir = get_shader_dir()
        files = [os.path.join(shader_dir, f) for f in list_shaders()]
        if files:
            return _format_size(sum(os.path.getsize(f) for f in files))
        return "0 B"
    else:
        return "-"

    if os.path.exists(path):
        return _format_size(os.path.getsize(path))
    return "-"


def _is_downloaded(key):
    entry = UPSCALE_REGISTRY.get(key, {})

    if entry.get("method") == "ml":
        return is_weight_downloaded(key)
    elif entry.get("method") == "onnx":
        return is_artcnn_downloaded(key)
    elif entry.get("method") == "shader":
        return is_anime4k_downloaded()
    return False


def models(
    delete: Optional[str] = typer.Option(None, "--delete", help="Delete a model by key"),
    download: Optional[str] = typer.Option(None, "--download", help="Download a model by key"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show file paths and hashes"),
    show_storage: bool = typer.Option(False, "--storage", help="Show storage directory"),
) -> None:
    """Manage upscaling model files.

    Without options, lists all models from the registry with download status.
    Use --delete to remove a model, --download to fetch one.
    """
    banner("models")

    if show_storage:
        console.print(f"  Models:  [dim]{WEIGHTS_DIR}[/dim]")
        console.print(f"  Shaders: [dim]{os.path.join(get_models_dir(), 'anime4k')}[/dim]")
        console.print(f"  ONNX:    [dim]{os.path.join(get_models_dir(), 'artcnn')}[/dim]")
        console.print(f"  FFmpeg:  [dim]{os.path.dirname(os.path.dirname(WEIGHTS_DIR))}/ffmpeg/bin[/dim]")
        return

    if delete:
        if delete in get_ml_models():
            path = get_weight_path(delete)
            if os.path.exists(path):
                os.unlink(path)
                ok(f"Deleted: {delete}")
            else:
                fail(f"Not on disk: {delete}")
        elif delete == "anime4k":
            import glob
            shader_dir = get_shader_dir()
            deleted = 0
            if os.path.exists(shader_dir):
                for fp in glob.glob(os.path.join(shader_dir, "*.glsl")):
                    os.unlink(fp)
                    deleted += 1
            ok(f"Deleted {deleted} shader files")
        elif delete in get_onnx_models():
            path = get_artcnn_path(delete)
            if os.path.exists(path):
                os.unlink(path)
                ok(f"Deleted: {delete}")
            else:
                fail(f"Not on disk: {delete}")
        else:
            fail(f"Unknown key: {delete}")
            raise typer.Exit(1)
        return

    if download:
        if download in get_ml_models():
            entry = get_ml_models()[download]
            console.print(f"  Downloading [accent]{entry.get('name', download)}[/accent]...")
            success = download_weights(download)
            ok(f"Downloaded: {download}") if success else fail(f"Failed: {download}")
        elif download == "anime4k":
            console.print("  Downloading [accent]Anime4K shaders[/accent]...")
            shaders = download_anime4k_shaders()
            ok(f"Downloaded {len(shaders)} shader files")
        elif download in get_onnx_models():
            entry = get_onnx_models()[download]
            console.print(f"  Downloading [accent]{entry.get('name', download)}[/accent]...")
            download_artcnn(download)
            ok(f"Downloaded: {download}")
        else:
            fail(f"Unknown key: {download}")
            raise typer.Exit(1)
        return

    if verbose:
        columns = ("Key", "bright_black", {}), ("File", "bright_black", {}), ("Size", "bright_black", {}), ("Hash", "bright_black", {}), ("Status", "bright_black", {})
    else:
        columns = ("Model", "bright_black", {}), ("Method", "bright_black", {"width": 8}), ("Size", "bright_black", {}), ("", "bright_black", {})

    table = make_table(*columns, title=None)

    for key, entry in UPSCALE_REGISTRY.items():
        size = _get_model_size(key)
        downloaded = _is_downloaded(key)
        method_tag = {"ml": "[accent]ml[/]", "shader": "[#facc15]shader[/]", "onnx": "[#60a5fa]onnx[/]"}.get(entry["method"], entry["method"])
        status = "[accent]downloaded[/]" if downloaded else "[muted]not downloaded[/]"

        if verbose:
            if entry.get("method") == "ml":
                file = entry.get("file", "-")
                hash_val = (entry.get("hash", "")[:12] + "...") if entry.get("hash") else "-"
            elif entry.get("method") == "onnx":
                file = entry.get("file", "-")
                hash_val = "-"
            else:
                file = "shaders.zip"
                hash_val = "-"
            table.add_row(f"[bold]{key}[/]", file, size, hash_val, status)
        else:
            table.add_row(f"[bold]{entry['name']}[/]", method_tag, size, status)

    console.print(table)

    console.print()
    total = sum(1 for k in UPSCALE_REGISTRY if _is_downloaded(k))
    console.print(f"  [dim]{total}/{len(UPSCALE_REGISTRY)} models downloaded[/]  "
                  f"[dim]--download <key> | --delete <key> | --verbose | --storage[/]")
