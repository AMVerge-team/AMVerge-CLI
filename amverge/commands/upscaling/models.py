from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import typer

from ...ui import banner, console, make_table, ok, fail
from ...core.infra.config import get_models_dir
from ...core.upscaling.registry import (
    UPSCALE_REGISTRY, get_ml_models, get_onnx_models,
)
from ...core.upscaling.weight_loader import (
    WEIGHTS_DIR, get_weight_path, is_weight_downloaded as _upscale_is_downloaded, download_weights as _upscale_download,
)
from ...core.upscaling.anime4k import (
    download_anime4k_shaders, is_anime4k_downloaded as _anime4k_is_downloaded, list_shaders, get_shader_dir,
)
from ...core.upscaling.artcnn import (
    download_artcnn, is_artcnn_downloaded as _artcnn_is_downloaded, get_artcnn_path,
)
from ...core.interpolation.registry import INTERPOLATION_REGISTRY
from ...core.interpolation.weight_loader import (
    is_weight_downloaded as _interp_is_downloaded,
    download_weights as _interp_download,
    get_weight_path as _interp_weight_path,
)
from ...core.interpolation.flowframes import FLOWFRAMES_MODELS, is_flowframes_model_installed, flowframes_available


def _format_size(size_bytes):
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    if size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes} B"


def _upscale_model_size(key):
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


def _upscale_is_downloaded_check(key):
    entry = UPSCALE_REGISTRY.get(key, {})

    if entry.get("method") == "ml":
        return _upscale_is_downloaded(key)
    elif entry.get("method") == "onnx":
        return _artcnn_is_downloaded(key)
    elif entry.get("method") == "shader":
        return _anime4k_is_downloaded()
    return False


def models(
    upscale_only: bool = typer.Option(False, "--upscale", "-u", help="Show only upscale models"),
    interpolation_only: bool = typer.Option(False, "--interpolation", "-i", help="Show only interpolation models"),
    flowframes_only: bool = typer.Option(False, "--flowframes", "-ff", help="Show only Flowframes models"),
    delete: Optional[str] = typer.Option(None, "--delete", help="Delete a model by key"),
    download: Optional[str] = typer.Option(None, "--download", help="Download a model by key"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show file paths and hashes"),
    show_storage: bool = typer.Option(False, "--storage", help="Show storage directories"),
) -> None:
    """Manage model files (upscaling and interpolation).

    Without options, lists all models from both registries with download status.
    Use --upscale, --interpolation, or --flowframes to filter. Use --delete to remove a model,
    --download to fetch one.
    """
    banner("models")

    if show_storage:
        console.print(f"  Upscale:       [dim]{WEIGHTS_DIR}[/dim]")
        console.print(f"  Anime4K:       [dim]{os.path.join(get_models_dir(), 'anime4k')}[/dim]")
        console.print(f"  ArtCNN ONNX:   [dim]{os.path.join(get_models_dir(), 'artcnn')}[/dim]")
        console.print(f"  Interpolation: [dim]{os.path.join(get_models_dir(), 'interpolation')}[/dim]")
        console.print(f"  Flowframes:    [dim]%LOCALAPPDATA%\\Flowframes\\FlowframesData\\pkgs\\[/dim]")
        console.print(f"  FFmpeg:        [dim]{os.path.dirname(os.path.dirname(WEIGHTS_DIR))}/ffmpeg/bin[/dim]")
        return

    show_both = not upscale_only and not interpolation_only and not flowframes_only

    if delete:
        _handle_delete(delete, upscale_only, interpolation_only, flowframes_only, show_both)
        return

    if download:
        _handle_download_action(download, upscale_only, interpolation_only, flowframes_only, show_both)
        return

    if show_both or upscale_only:
        _show_upscale_table(verbose)
    if show_both or interpolation_only:
        if show_both or upscale_only:
            console.print()
        _show_interpolation_table(verbose)
    if show_both or flowframes_only:
        if show_both or upscale_only or interpolation_only:
            console.print()
        _show_flowframes_table(verbose)


def _handle_delete(key, upscale_only, interpolation_only, flowframes_only, show_both):
    in_upscale = key in UPSCALE_REGISTRY or key in get_ml_models() or key in get_onnx_models() or key == "anime4k"
    in_interp = key in INTERPOLATION_REGISTRY
    in_ff = key in FLOWFRAMES_MODELS

    if not in_upscale and not in_interp and not in_ff:
        fail(f"Unknown key: {key}")
        raise typer.Exit(1)

    if in_upscale and (show_both or upscale_only):
        _do_upscale_delete(key)
    if in_interp and (show_both or interpolation_only):
        _do_interp_delete(key)
    if in_ff:
        fail("Flowframes models are managed by Flowframes.exe, not amverge")


def _do_upscale_delete(key):
    if key in get_ml_models():
        path = get_weight_path(key)
        if os.path.exists(path):
            os.unlink(path)
            ok(f"Deleted: {key}")
        else:
            fail(f"Not on disk: {key}")
    elif key == "anime4k":
        import glob
        shader_dir = get_shader_dir()
        deleted = 0
        if os.path.exists(shader_dir):
            for fp in glob.glob(os.path.join(shader_dir, "*.glsl")):
                os.unlink(fp)
                deleted += 1
        ok(f"Deleted {deleted} shader files")
    elif key in get_onnx_models():
        path = get_artcnn_path(key)
        if os.path.exists(path):
            os.unlink(path)
            ok(f"Deleted: {key}")
        else:
            fail(f"Not on disk: {key}")


def _do_interp_delete(key):
    path = _interp_weight_path(key)
    if os.path.exists(path):
        os.unlink(path)
        ok(f"Deleted: {key}")
    else:
        fail(f"Not on disk: {key}")


def _handle_download_action(key, upscale_only, interpolation_only, flowframes_only, show_both):
    in_upscale = key in get_ml_models() or key == "anime4k" or key in get_onnx_models()
    in_interp = key in INTERPOLATION_REGISTRY
    in_ff = key in FLOWFRAMES_MODELS

    if not in_upscale and not in_interp and not in_ff:
        fail(f"Unknown key: {key}")
        raise typer.Exit(1)

    if in_upscale and (show_both or upscale_only):
        _do_upscale_download(key)
    if in_interp and (show_both or interpolation_only):
        _do_interp_download(key)
    if in_ff:
        fail("Flowframes models are managed by Flowframes.exe, not amverge")


def _do_upscale_download(key):
    if key in get_ml_models():
        entry = get_ml_models()[key]
        console.print(f"  Downloading [accent]{entry.get('name', key)}[/accent]...")
        success = _upscale_download(key)
        ok(f"Downloaded: {key}") if success else fail(f"Failed: {key}")
    elif key == "anime4k":
        console.print("  Downloading [accent]Anime4K shaders[/accent]...")
        shaders = download_anime4k_shaders()
        ok(f"Downloaded {len(shaders)} shader files")
    elif key in get_onnx_models():
        entry = get_onnx_models()[key]
        console.print(f"  Downloading [accent]{entry.get('name', key)}[/accent]...")
        download_artcnn(key)
        ok(f"Downloaded: {key}")


def _do_interp_download(key):
    entry = INTERPOLATION_REGISTRY[key]
    console.print(f"  Downloading [accent]{entry.get('name', key)}[/accent]...")
    _interp_download(key)
    ok(f"Downloaded: {key}")


def _show_upscale_table(verbose):
    if verbose:
        columns = ("Key", "bright_black", {}), ("File", "bright_black", {}), ("Size", "bright_black", {}), ("Hash", "bright_black", {}), ("Status", "bright_black", {})
    else:
        columns = ("Model", "bright_black", {}), ("Method", "bright_black", {"width": 8}), ("Size", "bright_black", {}), ("", "bright_black", {})

    table = make_table(*columns, title=None)

    for key, entry in UPSCALE_REGISTRY.items():
        size = _upscale_model_size(key)
        downloaded = _upscale_is_downloaded_check(key)
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
    total = sum(1 for k in UPSCALE_REGISTRY if _upscale_is_downloaded_check(k))
    console.print(f"  [dim]{total}/{len(UPSCALE_REGISTRY)} upscale models downloaded[/]  "
                  f"[dim]--download <key> | --delete <key> | --verbose | --storage[/]")


def _show_interpolation_table(verbose):
    if verbose:
        columns = ("Key", "bright_black", {}), ("File", "bright_black", {}), ("Size", "bright_black", {}), ("Hash", "bright_black", {}), ("Status", "bright_black", {})
    else:
        columns = ("Model", "bright_black", {}), ("Method", "bright_black", {"width": 10}), ("Size", "bright_black", {}), ("", "bright_black", {})

    table = make_table(*columns, title=None)

    for key, entry in INTERPOLATION_REGISTRY.items():
        downloaded = _interp_is_downloaded(key)
        path = _interp_weight_path(key)
        size = _format_size(os.path.getsize(path)) if os.path.exists(path) else "-"
        method_tag = "[#a78bfa]rife[/]"
        status = "[accent]downloaded[/]" if downloaded else "[muted]not downloaded[/]"

        if verbose:
            file = entry.get("file", "-")
            hash_val = (entry.get("hash", "")[:12] + "...") if entry.get("hash") else "-"
            table.add_row(f"[bold]{key}[/]", file, size, hash_val, status)
        else:
            table.add_row(f"[bold]{entry['name']}[/]", method_tag, size, status)

    console.print(table)
    console.print()
    total = sum(1 for k in INTERPOLATION_REGISTRY if _interp_is_downloaded(k))
    console.print(f"  [dim]{total}/{len(INTERPOLATION_REGISTRY)} interpolation models downloaded[/]  "
                  f"[dim]--download <key> | --delete <key> | --verbose | --storage[/]")


def _show_flowframes_table(verbose):
    if not flowframes_available:
        return

    if verbose:
        columns = ("Key", "bright_black", {}), ("Engine", "bright_black", {}), ("Model", "bright_black", {}), ("", "bright_black", {})
    else:
        columns = ("Model", "bright_black", {}), ("Engine", "bright_black", {"width": 14}), ("", "bright_black", {})

    table = make_table(*columns, title=None)

    for key, entry in FLOWFRAMES_MODELS.items():
        installed = is_flowframes_model_installed(key)
        engine_tag = "[#a78bfa]rife[/]" if entry["engine"].startswith("Rife") else f"[#facc15]{entry['engine']}[/]"
        status = "[accent]installed[/]" if installed else "[muted]not installed[/]"

        if verbose:
            table.add_row(f"[bold]{key}[/]", entry["engine"], entry["model_name"], status)
        else:
            table.add_row(f"[bold]{entry['name']}[/]", engine_tag, status)

    console.print(table)
    console.print()
    total = sum(1 for k in FLOWFRAMES_MODELS if is_flowframes_model_installed(k))
    console.print(f"  [dim]{total}/{len(FLOWFRAMES_MODELS)} Flowframes models installed[/]  "
                  f"[dim]managed by Flowframes.exe | --verbose | --storage[/]")
