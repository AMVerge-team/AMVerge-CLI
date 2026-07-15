"""
Chain multiple operations (deadframes, upscale, interpolate) into a pipeline.

Run interactively with no arguments. Load saved presets with --load <name>.
"""
from __future__ import annotations

from pathlib import Path

import typer

from ...ui import banner, console, err, make_progress, ok, fail, warn
from ...core.pipeline.presets import list_presets, load_preset, save_preset, delete_preset


def _prompt_deadframes(defaults: dict | None = None):
    from ...ui import err as econsole
    from rich.prompt import Prompt, Confirm

    d = defaults or {}
    banner("deadframes settings")
    econsole.print("  [muted]Configure deadframe removal options.[/]\n")

    auto = d.get("auto", True)
    keep_talking = d.get("keep_talking", False)
    keep_camera = d.get("keep_camera", False)
    safe = d.get("safe", False)
    cadence = d.get("cadence", 3)
    detect_scale = d.get("detect_scale", 1.0)
    small_movements = d.get("small_movements")
    prores = d.get("prores", False)
    no_audio = d.get("no_audio", False)
    parallax = d.get("parallax", False)

    auto_val = Confirm.ask(
        "  [accent]>[/]  [label]Auto-calibrate thresholds?[/]",
        console=econsole, default=auto,
    )
    safe_val = Confirm.ask(
        "  [accent]>[/]  [label]Safe mode (keep-talking + keep-camera)?[/]",
        console=econsole, default=safe,
    )
    if not safe_val:
        keep_talking_val = Confirm.ask(
            "  [accent]>[/]  [label]Keep talking (subtle mouth motion)?[/]",
            console=econsole, default=keep_talking,
        )
        keep_camera_val = Confirm.ask(
            "  [accent]>[/]  [label]Keep camera pan/zoom/shake?[/]",
            console=econsole, default=keep_camera,
        )
    else:
        keep_talking_val = True
        keep_camera_val = True

    cadence_val = int(Prompt.ask(
        "  [accent]>[/]  [label]Cadence (min consecutive dead to drop)[/]",
        console=econsole, default=str(cadence),
    ) or cadence)

    no_audio_val = Confirm.ask(
        "  [accent]>[/]  [label]Drop audio?[/]",
        console=econsole, default=no_audio,
    )
    prores_val = Confirm.ask(
        "  [accent]>[/]  [label]ProRes output?[/]",
        console=econsole, default=prores,
    )
    parallax_val = Confirm.ask(
        "  [accent]>[/]  [label]Parallax mode (invert camera motion rule)?[/]",
        console=econsole, default=parallax,
    )

    return {
        "auto": auto_val,
        "keep_talking": keep_talking_val,
        "keep_camera": keep_camera_val,
        "safe": safe_val,
        "cadence": cadence_val,
        "detect_scale": detect_scale,
        "small_movements": small_movements,
        "prores": prores_val,
        "no_audio": no_audio_val,
        "parallax": parallax_val,
    }


def _prompt_upscale(defaults: dict | None = None):
    from ...ui import err as econsole
    from rich.prompt import Prompt, Confirm

    d = defaults or {}
    banner("upscale settings")
    econsole.print("  [muted]Configure AI upscaling options.[/]\n")

    from ...core.upscaling.registry import UPSCALE_REGISTRY, get_all_model_keys

    keys = get_all_model_keys()
    model = d.get("model", keys[0] if keys else "adore")
    scale = d.get("scale", 2)
    preset = d.get("preset", "high")

    econsole.print("  [muted]Available models:[/]")
    for k, e in UPSCALE_REGISTRY.items():
        econsole.print(f"    [accent]{k}[/] - {e.get('name', k)} [{e.get('method', '?')}]")
    econsole.print()

    model_val = Prompt.ask(
        "  [accent]>[/]  [label]Model key[/]",
        console=econsole, default=model,
    ) or model
    if model_val not in UPSCALE_REGISTRY:
        fail(f"Unknown model: {model_val}")
        model_val = keys[0] if keys else model

    from ...core.upscaling.registry import get_model_scales
    valid_scales = get_model_scales(model_val)
    scale_val = int(Prompt.ask(
        f"  [accent]>[/]  [label]Scale factor[/] [muted]({'/'.join(str(s) for s in valid_scales)})[/]",
        console=econsole, default=str(scale),
    ) or scale)

    preset_val = Prompt.ask(
        "  [accent]>[/]  [label]Quality preset[/] [muted](archival/high/balanced/fast/draft)[/]",
        console=econsole, default=preset,
    ) or preset

    return {
        "model": model_val,
        "scale": scale_val,
        "preset": preset_val,
    }


def _prompt_interpolate(defaults: dict | None = None):
    from ...ui import err as econsole
    from rich.prompt import Prompt, Confirm

    d = defaults or {}
    banner("interpolation settings")
    econsole.print("  [muted]Configure frame interpolation options.[/]\n")

    from ...core.interpolation import INTERPOLATION_REGISTRY

    keys = list(INTERPOLATION_REGISTRY.keys())
    model = d.get("model", "rife4.25")
    factor = d.get("factor", 2)
    preset = d.get("preset", "high")

    econsole.print("  [muted]Available models:[/]")
    for k, e in INTERPOLATION_REGISTRY.items():
        econsole.print(f"    [accent]{k}[/] - {e.get('name', k)}")
    econsole.print()

    model_val = Prompt.ask(
        "  [accent]>[/]  [label]Model key[/]",
        console=econsole, default=model,
    ) or model
    if model_val not in INTERPOLATION_REGISTRY:
        fail(f"Unknown model: {model_val}")
        model_val = "rife4.25"

    factor_val = int(Prompt.ask(
        "  [accent]>[/]  [label]Frame multiplier[/] [muted](2-64)[/]",
        console=econsole, default=str(factor),
    ) or factor)

    preset_val = Prompt.ask(
        "  [accent]>[/]  [label]Quality preset[/] [muted](archival/high/balanced/fast/draft)[/]",
        console=econsole, default=preset,
    ) or preset

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

    Run interactively to configure each step. Save presets with --save <name>.
    Load saved presets with --load <name>.
    """
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
        from rich.prompt import Prompt
        raw = Prompt.ask(
            "  [accent]>[/]  [label]Input video path[/]",
            console=err,
        )
        if not raw:
            fail("Input video is required.")
            raise typer.Exit(1)
        input = Path(raw)
    if not input.exists():
        fail(f"File not found: {input}")
        raise typer.Exit(1)

    from rich.prompt import Prompt, Confirm
    from rich.panel import Panel

    if not load:
        banner("pipeline")
        err.print("  [muted]Select operations to run in sequence.[/]\n")

        ops = config.get("operations", [])
        use_deadframes = Confirm.ask(
            "  [accent]>[/]  [label]Remove deadframes?[/]",
            console=err, default=("deadframes" in ops),
        )
        use_upscale = Confirm.ask(
            "  [accent]>[/]  [label]AI upscale?[/]",
            console=err, default=("upscale" in ops),
        )
        use_interpolate = Confirm.ask(
            "  [accent]>[/]  [label]Frame interpolation?[/]",
            console=err, default=("interpolate" in ops),
        )

        if not use_deadframes and not use_upscale and not use_interpolate:
            fail("At least one operation must be selected.")
            raise typer.Exit(1)

        config["operations"] = []
        if use_deadframes:
            config["operations"].append("deadframes")
            config["deadframes"] = _prompt_deadframes(config.get("deadframes"))
        if use_upscale:
            config["operations"].append("upscale")
            config["upscale"] = _prompt_upscale(config.get("upscale"))
        if use_interpolate:
            config["operations"].append("interpolate")
            config["interpolate"] = _prompt_interpolate(config.get("interpolate"))

    from ...core.infra.ffmpeg_bootstrap import is_portable_ffmpeg_installed, ensure_ffmpeg
    if not is_portable_ffmpeg_installed():
        console.print("  [warn]FFmpeg not found.[/warn]")
        if yes or Confirm.ask("  Download portable FFmpeg?", console=err, default=True):
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
        if Confirm.ask("  [accent]>[/]  [label]Save these settings as a preset?[/]", console=err, default=False):
            save_name = Prompt.ask(
                "  [accent]>[/]  [label]Preset name[/]",
                console=err, default="my_pipeline",
            )
            if save_name:
                save_preset(save_name, config)
                ok(f"Saved preset: {save_name}")

    if save and not load:
        save_preset(save, config)
        ok(f"Saved preset: {save}")

    if not Confirm.ask("  [accent]>[/]  [label]Run pipeline?[/]", console=err, default=True):
        return

    temp_dir = input.parent
    current_input = str(input.resolve())

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

    console.print(f"\n  [accent]{'─' * 40}[/]")
    ok(f"Pipeline complete: {current_input}")
