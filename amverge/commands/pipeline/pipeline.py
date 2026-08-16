"""
Chain multiple operations (deadframes, upscale, interpolate) into a pipeline.

Requires at least 2 of [deadframes], [upscale], [interpolation] extras installed.
Run interactively with no arguments. Load saved presets with --load <name>.
"""
from __future__ import annotations

from functools import cache
from pathlib import Path

import typer

from ...ui import banner, console, err, make_progress, ok, fail, warn
from ...ui.interactive import select, checkboxes, confirm, text_input
from ...core.pipeline.presets import list_presets, load_preset, save_preset, delete_preset


def _check_op_deadframes():
    try:
        from ...core.deadframes import DEADFRAMES_AVAILABLE as _ok
        return _ok
    except ImportError:
        return False


def _check_op_upscale():
    try:
        from ...core.upscaling import UPSCALE_AVAILABLE as _ok
        return _ok
    except ImportError:
        return False


def _check_op_interpolate():
    try:
        from ...core.interpolation import INTERPOLATION_AVAILABLE as _ok
        return _ok
    except ImportError:
        return False


@cache
def available_ops() -> dict[str, bool]:
    """Which pipeline operations have their extras installed.

    Computed on first call rather than at import. Each _check_op_* import pulls
    an ML stack, and cli.py imports this module for every command — so doing it
    at module scope made `amverge --help` load torch. @cache keeps the
    compute-once semantics the module constant used to have.

    NOTE: a module-level __getattr__ cannot replace this, because it does not
    fire for same-module global lookups (the reads inside pipeline() below).
    """
    return {
        "deadframes": _check_op_deadframes(),
        "upscale": _check_op_upscale(),
        "interpolate": _check_op_interpolate(),
    }


def pipeline_enabled(ops: dict[str, bool] | None = None) -> bool:
    """A pipeline needs at least two operations available to be worth running."""
    ops = available_ops() if ops is None else ops
    return sum(1 for v in ops.values() if v) >= 2


def _prompt_deadframes(defaults: dict | None = None):
    d = defaults or {}
    banner("deadframes settings")
    err.print("  [muted]Configure deadframe removal options.[/]\n")

    auto = d.get("auto", True)
    safe = d.get("safe", False)
    keep_talking = d.get("keep_talking", False)
    keep_camera = d.get("keep_camera", False)
    cadence = d.get("cadence", 3)
    no_audio = d.get("no_audio", False)
    prores = d.get("prores", False)
    parallax = d.get("parallax", False)

    auto_val = confirm("Auto-calibrate thresholds?", default=auto)
    err.print()

    safe_val = confirm("Safe mode (keep-talking + keep-camera)?", default=safe)
    err.print()

    if not safe_val:
        keep_talking_val = confirm("Keep talking (subtle mouth motion)?", default=keep_talking)
        err.print()
        keep_camera_val = confirm("Keep camera pan/zoom/shake?", default=keep_camera)
        err.print()
    else:
        keep_talking_val = True
        keep_camera_val = True

    cadence_val = text_input("Cadence (min consecutive dead to drop)", str(cadence))
    try:
        cadence_val = int(cadence_val) if cadence_val else cadence
    except ValueError:
        cadence_val = cadence

    no_audio_val = confirm("Drop audio?", default=no_audio)
    err.print()
    prores_val = confirm("ProRes output?", default=prores)
    err.print()
    parallax_val = confirm("Parallax mode (invert camera motion rule)?", default=parallax)
    err.print()

    return {
        "auto": auto_val,
        "keep_talking": keep_talking_val,
        "keep_camera": keep_camera_val,
        "safe": safe_val,
        "cadence": cadence_val,
        "detect_scale": d.get("detect_scale", 1.0),
        "small_movements": d.get("small_movements"),
        "prores": prores_val,
        "no_audio": no_audio_val,
        "parallax": parallax_val,
    }


def _prompt_upscale(defaults: dict | None = None):
    from ...core.upscaling.registry import UPSCALE_REGISTRY, get_all_model_keys, get_model_scales

    d = defaults or {}
    banner("upscale settings")
    err.print("  [muted]Configure AI upscaling options.[/]\n")

    keys = get_all_model_keys()
    default_key = d.get("model", keys[0] if keys else "adore")
    default_idx = keys.index(default_key) if default_key in keys else 0

    model_val = select(
        [f"{k}  [dim]{UPSCALE_REGISTRY[k].get('name', k)} [muted]({UPSCALE_REGISTRY[k].get('method', '?')})[/]" for k in keys],
        "Select model:",
        default_idx,
        console=err,
    )
    if model_val < 0:
        fail("Cancelled")
        raise typer.Exit(1)
    model_val = keys[model_val]

    scale = d.get("scale", 2)
    valid_scales = get_model_scales(model_val)
    scale_options = [f"{s}x" for s in valid_scales]
    scale_default = valid_scales.index(scale) if scale in valid_scales else 0
    scale_idx = select(scale_options, "Select scale factor:", scale_default, console=err)
    if scale_idx < 0:
        fail("Cancelled")
        raise typer.Exit(1)
    scale_val = valid_scales[scale_idx]

    preset = d.get("preset", "high")
    preset_options = ["archival", "high", "balanced", "fast", "draft"]
    preset_default = preset_options.index(preset) if preset in preset_options else 1
    preset_idx = select(preset_options, "Select quality preset:", preset_default, console=err)
    if preset_idx < 0:
        fail("Cancelled")
        raise typer.Exit(1)
    preset_val = preset_options[preset_idx]

    return {
        "model": model_val,
        "scale": scale_val,
        "preset": preset_val,
    }


def _prompt_interpolate(defaults: dict | None = None):
    from ...core.interpolation import INTERPOLATION_REGISTRY

    d = defaults or {}
    banner("interpolation settings")
    err.print("  [muted]Configure frame interpolation options.[/]\n")

    keys = list(INTERPOLATION_REGISTRY.keys())
    default_key = d.get("model", "rife4.25")
    default_idx = keys.index(default_key) if default_key in keys else 0

    model_val = select(
        [f"{k}  [dim]{INTERPOLATION_REGISTRY[k].get('name', k)}[/]" for k in keys],
        "Select model:",
        default_idx,
        console=err,
    )
    if model_val < 0:
        fail("Cancelled")
        raise typer.Exit(1)
    model_val = keys[model_val]

    factor = d.get("factor", 2)
    factor_options = [f"{i}x" for i in [2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 64]]
    factor_default = 0
    for i, f in enumerate(factor_options):
        if int(f.rstrip("x")) == factor:
            factor_default = i
            break
    factor_idx = select(factor_options, "Select frame multiplier:", factor_default, console=err)
    if factor_idx < 0:
        fail("Cancelled")
        raise typer.Exit(1)
    factor_val = int(factor_options[factor_idx].rstrip("x"))

    preset = d.get("preset", "high")
    preset_options = ["archival", "high", "balanced", "fast", "draft"]
    preset_default = preset_options.index(preset) if preset in preset_options else 1
    preset_idx = select(preset_options, "Select quality preset:", preset_default, console=err)
    if preset_idx < 0:
        fail("Cancelled")
        raise typer.Exit(1)
    preset_val = preset_options[preset_idx]

    return {
        "model": model_val,
        "factor": factor_val,
        "preset": preset_val,
    }


def _run_step_deadframes(input_path: str, output_path: str, config: dict, progress_cb):
    from ...core.deadframes import DEADFRAMES_AVAILABLE
    if not DEADFRAMES_AVAILABLE:
        fail("OpenCV not installed. Run: pip install amverge[deadframes]")
        raise typer.Exit(1)
    from ...core.deadframes.engine import run_deadframes
    from ...core.deadframes.registry import DEADFRAMES_REGISTRY
    method = config.get("method", "heuristic")
    if method not in DEADFRAMES_REGISTRY:
        fail(f"Unknown deadframes method: {method}")
        method = "heuristic"
    from ...core.deadframes import is_weight_downloaded, download_weights
    entry = DEADFRAMES_REGISTRY[method]
    if "file" in entry and not is_weight_downloaded(method):
        console.print(f"  Downloading [accent]{entry['name']}[/accent]...")
        download_weights(method)
    use_keep_talking = config.get("keep_talking", False) or config.get("safe", False)
    use_keep_camera = config.get("keep_camera", False) or config.get("safe", False)
    auto_mode = config.get("auto", False) or config.get("safe", False) or use_keep_talking or use_keep_camera
    return run_deadframes(
        input_path=input_path,
        output_path=output_path,
        keep_talking=use_keep_talking,
        keep_camera=use_keep_camera,
        parallax_mode=config.get("parallax", False),
        auto=auto_mode,
        cadence=config.get("cadence", 3),
        detect_scale=config.get("detect_scale", 1.0),
        small_movements=config.get("small_movements"),
        prores=config.get("prores", False),
        no_audio=config.get("no_audio", False),
        progress_cb=progress_cb,
    )


def _run_step_upscale(input_path: str, output_path: str, config: dict, progress_cb):
    from ...core.upscaling import UPSCALE_AVAILABLE
    if not UPSCALE_AVAILABLE:
        fail("Upscaling not available. Run: pip install amverge[upscale]")
        raise typer.Exit(1)
    from ...core.upscaling import (
        upscale_model, download_weights, is_weight_downloaded,
        UPSCALE_REGISTRY,
    )
    model = config.get("model", "adore")
    entry = UPSCALE_REGISTRY.get(model)
    if entry is None:
        fail(f"Unknown upscale model: {model}")
        raise typer.Exit(1)
    method = entry.get("method", "ml")
    if method == "ml":
        if not is_weight_downloaded(model):
            console.print(f"  Downloading [accent]{entry.get('name', model)}[/accent]...")
            download_weights(model, progress_cb=lambda p, m: None)
    elif method == "shader":
        from ...core.upscaling.anime4k import is_anime4k_downloaded, download_anime4k_shaders
        if not is_anime4k_downloaded():
            console.print("  Downloading Anime4K shaders...")
            download_anime4k_shaders()
    elif method == "onnx":
        from ...core.upscaling.artcnn import is_artcnn_downloaded, download_artcnn
        if not is_artcnn_downloaded(model):
            download_artcnn(model)
    upscale_model(
        model_key=model,
        input_path=input_path,
        output_path=output_path,
        scale=config.get("scale", 2),
        preset=config.get("preset", "high"),
        progress_cb=progress_cb,
    )


def _run_step_interpolate(input_path: str, output_path: str, config: dict, progress_cb):
    from ...core.interpolation import INTERPOLATION_AVAILABLE
    if not INTERPOLATION_AVAILABLE:
        fail("Interpolation not available. Run: pip install amverge[interpolation]")
        raise typer.Exit(1)
    from ...core.interpolation import (
        interpolate_video, download_weights, is_weight_downloaded,
        INTERPOLATION_REGISTRY,
    )
    model = config.get("model", "rife4.25")
    entry = INTERPOLATION_REGISTRY.get(model)
    if entry is None:
        fail(f"Unknown interpolation model: {model}")
        raise typer.Exit(1)
    if not is_weight_downloaded(model):
        console.print(f"  Downloading [accent]{entry.get('name', model)}[/accent]...")
        download_weights(model, progress_cb=lambda p, m: None)
    interpolate_video(
        input_path=input_path,
        output_path=output_path,
        model_key=model,
        factor=config.get("factor", 2),
        preset=config.get("preset", "high"),
        progress_cb=progress_cb,
    )


def pipeline(
    input: Path = typer.Argument(None, help="Input video file"),
    load: str = typer.Option(None, "--load", "-l", help="Load a saved pipeline preset by name"),
    save: str = typer.Option(None, "--save", "-s", help="Save current settings as a named preset"),
    list_presets_flag: bool = typer.Option(False, "--list", help="List saved pipeline presets"),
    delete: str = typer.Option(None, "--delete", help="Delete a saved pipeline preset"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Auto-confirm downloads"),
) -> None:
    """Chain multiple operations (deadframes, upscale, interpolate) into a pipeline.

    Requires at least 2 of the deadframes, upscale, interpolation extras installed.
    Run interactively with arrow-key navigation. Save presets with --save <name>.
    Load saved presets with --load <name>.
    """
    ops = available_ops()
    if not pipeline_enabled(ops):
        available = [k for k, v in ops.items() if v]
        missing_ops = [k for k, v in ops.items() if not v]
        fail(
            f"Pipeline requires at least 2 operations installed. "
            f"You have: {', '.join(available) if available else 'none'}. "
            f"Missing: {', '.join(missing_ops)}. "
            f"Install extras: pip install amverge[{' '.join(missing_ops)}]"
        )
        raise typer.Exit(1)

    if list_presets_flag:
        banner("pipeline presets")
        presets = list_presets()
        if not presets:
            console.print("  [muted]No saved presets.[/]")
        else:
            for p in presets:
                cfg = load_preset(p)
                ops = cfg.get("operations", []) if cfg else []
                console.print(f"  [accent]{p}[/]  [muted]{', '.join(ops)}[/]")
        console.print()
        return

    if delete:
        if delete_preset(delete):
            ok(f"Deleted preset: {delete}")
        else:
            fail(f"Preset not found: {delete}")
        return

    config = {"operations": []}
    if load:
        loaded = load_preset(load)
        if loaded is None:
            fail(f"Preset not found: {load}")
            raise typer.Exit(1)
        config = loaded
        console.print(f"  Loaded preset: [accent]{load}[/]")
        console.print(f"  Operations: [accent]{', '.join(config.get('operations', []))}[/]")

    if input is None:
        raw = text_input("Input video path", "")
        if not raw:
            fail("Input video is required.")
            raise typer.Exit(1)
        input = Path(raw)
    if not input.exists():
        fail(f"File not found: {input}")
        raise typer.Exit(1)

    if not load:
        banner("pipeline")
        err.print("  [muted]Select operations to run in sequence. Use space to toggle, enter to confirm.[/]\n")

        op_names = [
            f"{k}  [dim]{lbl}[/]"
            for k, lbl in [
                ("deadframes", "(remove static frames)"),
                ("upscale", "(AI super-resolution)"),
                ("interpolate", "(frame interpolation)"),
            ]
            if ops.get(k)
        ]
        op_keys_selected = [k for k in op_names]
        for i, name in enumerate(op_names):
            for k in ["deadframes", "upscale", "interpolate"]:
                if name.startswith(k):
                    op_keys_selected[i] = k
                    break
        ops_defaults = [
            i for i, k in enumerate(op_keys_selected)
            if k in config.get("operations", [])
        ]
        selected = checkboxes(op_names, "Select operations:", defaults=ops_defaults, console=err)
        if not selected:
            fail("At least one operation required.")
            raise typer.Exit(1)

        config["operations"] = [op_keys_selected[i] for i in selected]
        err.print()

        if "deadframes" in config["operations"]:
            config["deadframes"] = _prompt_deadframes(config.get("deadframes"))
        if "upscale" in config["operations"]:
            config["upscale"] = _prompt_upscale(config.get("upscale"))
        if "interpolate" in config["operations"]:
            config["interpolate"] = _prompt_interpolate(config.get("interpolate"))

    from ...core.infra.ffmpeg_bootstrap import is_portable_ffmpeg_installed, ensure_ffmpeg
    if not is_portable_ffmpeg_installed():
        console.print("  [warn]FFmpeg not found.[/warn]")
        if confirm("Download portable FFmpeg?", default=True):
            with make_progress() as progress:
                tid = progress.add_task("Downloading FFmpeg...", total=100)
                ensure_ffmpeg(progress_cb=lambda p, m: progress.update(tid, completed=p, description=m))
            ok("FFmpeg installed")
        else:
            fail("FFmpeg is required.")
            raise typer.Exit(1)

    banner("pipeline summary")
    ops = config.get("operations", [])
    stem = input.stem
    console.print(f"  Input:     [dim]{input}[/]")
    console.print(f"  Pipeline:  [accent]{' -> '.join(ops)}[/]")

    if "deadframes" in ops:
        df = config.get("deadframes", {})
        flags = []
        if df.get("safe"): flags.append("safe")
        if df.get("auto") and not df.get("safe"): flags.append("auto")
        if df.get("keep_talking") and not df.get("safe"): flags.append("talk")
        if df.get("keep_camera") and not df.get("safe"): flags.append("camera")
        if df.get("parallax"): flags.append("parallax")
        console.print(f"  Deadframes: [accent]{', '.join(flags) or 'default'}[/]")

    if "upscale" in ops:
        us = config.get("upscale", {})
        console.print(f"  Upscale:    [accent]{us.get('model', '?')}[/] "
                      f"x{us.get('scale', 2)} {us.get('preset', 'high')}")

    if "interpolate" in ops:
        ip = config.get("interpolate", {})
        console.print(f"  Interpolate: [accent]{ip.get('model', '?')}[/] "
                      f"x{ip.get('factor', 2)} {ip.get('preset', 'high')}")

    save_name = save
    if not save_name and not load:
        if confirm("Save these settings as a preset?", default=False):
            save_name = text_input("Preset name", "my_pipeline")
            if save_name:
                save_preset(save_name, config)
                ok(f"Saved preset: {save_name}")

    if save and not load:
        save_preset(save, config)
        ok(f"Saved preset: {save}")

    if not confirm("Run pipeline?", default=True):
        return

    temp_dir = input.parent
    current_input = str(input.resolve())
    original_input = current_input

    total_steps = len(ops)
    for idx, op in enumerate(ops):
        step_label = f"[{idx + 1}/{total_steps}] {op}"
        console.print(f"\n  [accent]{'─' * 40}[/]")
        console.print(f"  [accent bold]{step_label}[/]")

        if idx == total_steps - 1:
            out_name = f"{stem}_pipeline_{op}.mp4"
        else:
            out_name = f"{stem}_pipe_{op}.mp4"
        out_path = str(temp_dir / out_name)

        with make_progress() as progress:
            task_id = progress.add_task(f"Running {op}...", total=100)

            def _cb(pct, msg):
                progress.update(task_id, completed=pct, description=f"{op}: {msg}")

            try:
                if op == "deadframes":
                    _run_step_deadframes(current_input, out_path, config.get("deadframes", {}), _cb)
                elif op == "upscale":
                    _run_step_upscale(current_input, out_path, config.get("upscale", {}), _cb)
                elif op == "interpolate":
                    _run_step_interpolate(current_input, out_path, config.get("interpolate", {}), _cb)
            except Exception as e:
                fail(f"{op} failed: {e}")
                raise typer.Exit(1)

        ok(f"{op}: {out_path}")
        current_input = out_path

        if idx < total_steps - 1:
            next_op = ops[idx + 1]
            console.print()
            if not confirm(f"Use this output for [accent]{next_op}[/]? (No = revert to original)", default=True):
                current_input = original_input
                console.print(f"  [muted]Reverted to original input for {next_op}.[/]")

    console.print(f"\n  [accent]{'─' * 40}[/]")
    ok(f"Pipeline complete: {current_input}")
