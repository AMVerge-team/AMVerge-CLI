"""
Remove dead (static subject) frames from video using optical flow, feature
matching, and motion-area analysis.

Output is compacted CFR: kept frames packed back-to-back, duration shortens.
Requires: pip install amverge[deadframes]
"""
from __future__ import annotations

from pathlib import Path

import typer

from ...ui import banner, console, err, gpu_line, make_progress, ok, fail
from ...core.infra.diagnostics import get_gpu_info
from ...core.infra.ffmpeg_bootstrap import is_portable_ffmpeg_installed, ensure_ffmpeg
from ...core.upscaling.monitor import SystemMonitor, format_eta
from ...core.deadframes.registry import DEADFRAMES_REGISTRY


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
    from ...core.deadframes import is_weight_downloaded, download_weights

    if model_key == "heuristic":
        return

    entry = DEADFRAMES_REGISTRY.get(model_key, {})
    if "file" not in entry:
        return

    if is_weight_downloaded(model_key):
        return

    name = entry.get("name", model_key)
    console.print(f"  [warn]Model '{name}' not downloaded.[/warn]")
    if auto_yes or typer.confirm(f"  Download {name}?", default=True):
        with make_progress() as progress:
            task_id = progress.add_task(f"Downloading {name}...", total=100)

            def _cb(pct, msg):
                progress.update(task_id, completed=pct, description=msg)

            try:
                download_weights(model_key, progress_cb=_cb)
            except Exception as e:
                fail(f"Download failed for {name}: {e}")
                raise typer.Exit(1)
            ok(f"Model {name} downloaded")
    else:
        fail(f"Model {name} is required")
        raise typer.Exit(1)


def deadframes(
    input: Path = typer.Argument(None, help="Input video file"),
    output: Path = typer.Option(
        Path("deadframes_output.mp4"), "--output", "-o",
        help="Output video file",
    ),
    method: str = typer.Option(
        "heuristic", "--method", "-m",
        help="Detection method key from registry",
    ),
    auto: bool = typer.Option(
        False, "--auto",
        help="Auto-calibrate thresholds from frame-pair distribution",
    ),
    keep_talking: bool = typer.Option(
        False, "--keep-talking",
        help="Keep subtle dialogue/mouth motion (romance, talking heads)",
    ),
    keep_camera: bool = typer.Option(
        False, "--keep-camera",
        help="Keep camera pan/zoom/shake (vlogs, handheld)",
    ),
    safe: bool = typer.Option(
        False, "--safe",
        help="Safe mode: only drop completely static frames (--keep-talking --keep-camera)",
    ),
    cadence: int = typer.Option(
        3, "--cadence",
        help="Min consecutive dead frames to drop (preserves native animation holds)",
    ),
    detect_scale: float = typer.Option(
        1.0, "--detect-scale",
        help="Detection resolution scale (0.5 = half size, faster)",
    ),
    small_movements: float = typer.Option(
        None, "--small-movements",
        help="Custom flow threshold for small movements (0=keep all, 0.5=default)",
    ),
    prores: bool = typer.Option(
        False, "--prores",
        help="Export Apple ProRes instead of H.264",
    ),
    no_audio: bool = typer.Option(
        False, "--no-audio",
        help="Drop audio track",
    ),
    parallax: bool = typer.Option(
        False, "--parallax",
        help="Invert camera motion rule for parallax shots (foreground moves differently from background)",
    ),
    list_methods: bool = typer.Option(
        False, "--list-methods",
        help="List all available detection methods",
    ),
    credits: bool = typer.Option(
        False, "--credits",
        help="Show credits for deadframes technologies",
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y",
        help="Auto-confirm download prompts",
    ),
    download: bool = typer.Option(
        False, "--download",
        help="Download model weights without running",
    ),
    no_monitor: bool = typer.Option(
        False, "--no-monitor",
        help="Disable system monitor during processing",
    ),
) -> None:
    """Remove dead (static subject) frames from video and output compacted CFR.

    Detects frames where the main subject does not move meaningfully,
    drops them, and packs remaining frames back-to-back as constant
    frame rate. Safe to feed into frame interpolation (slow motion).

    Requires: pip install amverge[deadframes]
    """
    if list_methods:
        banner("deadframes methods")
        console.print()
        for key, entry in DEADFRAMES_REGISTRY.items():
            console.print(
                f"  [accent]{key}[/accent]  {entry['name']}  dim:{entry['method']}"
            )
            console.print(f"    {entry.get('description', '')}")
            console.print(f"    Credit: {entry.get('credit', '')}")
        console.print()
        return

    if credits:
        banner("deadframes credits")
        console.print()
        seen = set()
        for entry in DEADFRAMES_REGISTRY.values():
            cred = entry.get("credit", "")
            if cred and cred not in seen:
                console.print(f"  [accent]+[/accent] {cred}")
                seen.add(cred)
        console.print()
        return

    from ...core.deadframes import download_weights as _df_dl

    if download:
        if method not in DEADFRAMES_REGISTRY:
            fail(
                f"Unknown method '{method}'. Use --list-methods to see available methods."
            )
            raise typer.Exit(1)
        entry = DEADFRAMES_REGISTRY[method]
        if "file" not in entry:
            ok(f"'{method}' requires no download.")
            return
        console.print(f"  Downloading [accent]{entry['name']}[/accent]...")
        with make_progress() as progress:
            task_id = progress.add_task(
                f"Downloading {entry['name']}...", total=100
            )

            def _dl_cb(pct, msg):
                progress.update(task_id, completed=pct, description=msg)

            _df_dl(method, progress_cb=_dl_cb)
        ok(f"Downloaded: {method}")
        return

    if input is None:
        fail("Missing INPUT argument.")
        raise typer.Exit(1)
    if not input.exists():
        fail(f"File not found: {input}")
        raise typer.Exit(1)

    if method not in DEADFRAMES_REGISTRY:
        fail(
            f"Unknown method '{method}'. Use --list-methods to see available methods."
        )
        raise typer.Exit(1)

    output.parent.mkdir(parents=True, exist_ok=True)

    from ...core.deadframes.engine import DEADFRAMES_AVAILABLE

    if not DEADFRAMES_AVAILABLE:
        fail(
            "OpenCV (cv2) is required. Run: pip install amverge[deadframes]"
        )
        raise typer.Exit(1)

    _ensure_ffmpeg_interactive(auto_yes=yes)
    _ensure_model_downloaded(method, auto_yes=yes)

    use_keep_talking = keep_talking or safe
    use_keep_camera = keep_camera or safe
    auto_mode = auto or safe or keep_talking or keep_camera

    banner("deadframes")

    from ...core.video import get_video_info
    from ...core.deadframes.engine import _get_video_info as _get_vinfo

    info = get_video_info(str(input.resolve()))
    vinfo = _get_vinfo(str(input.resolve()))

    dur = info["duration"]
    dur_str_parts = []
    h, m, s = int(dur // 3600), int((dur % 3600) // 60), dur % 60
    if h:
        dur_str_parts.append(f"{h}h")
    if m:
        dur_str_parts.append(f"{m}m")
    dur_str_parts.append(f"{s:.1f}s")
    dur_str = " ".join(dur_str_parts)

    video_streams = [s for s in info.get("streams", []) if s.get("type") == "video"]
    vcodec = video_streams[0].get("codec", "?") if video_streams else "?"
    res = f"{vinfo['width']}x{vinfo['height']}" if vinfo.get("width") else "?"
    fps_val = vinfo.get("fps", info.get("fps", "?"))

    gpu_info = gpu_line(label="GPU:    ", torch_path=False)

    entry = DEADFRAMES_REGISTRY[method]
    console.print(f"  Method:  [accent]{entry['name']}[/accent]  "
                  f"Cadence: [accent]{cadence}[/accent]")
    console.print(f"  Video:   [accent]{vcodec}[/accent]  "
                  f"[accent]{res}[/accent]  "
                  f"[accent]{fps_val} fps[/accent]  "
                  f"[accent]{vinfo.get('pix_fmt', 'yuv420p')}[/accent]"
                  f"  {dur_str}")
    console.print(f"  Input:   [dim]{input}[/dim]")
    console.print(f"  Output:  [dim]{output}[/dim]")

    flags = []
    if auto_mode:
        flags.append("auto")
    if use_keep_talking:
        flags.append("talk")
    if use_keep_camera:
        flags.append("camera")
    if prores:
        flags.append("ProRes")
    if no_audio:
        flags.append("no-audio")
    if parallax:
        flags.append("parallax")
    if small_movements is not None:
        flags.append(f"small={small_movements}")
    if flags:
        console.print(f"  Flags:   [accent]{', '.join(flags)}[/accent]")

    from ...core.deadframes import run_deadframes

    monitor = SystemMonitor(enabled=not no_monitor)
    monitor.stats["gpu_name"] = gpu_info.get("gpu_name") or "GPU"
    monitor.start()

    def _update_display():
        from rich.live import Live
        from rich.panel import Panel
        from rich.progress import (
            Progress,
            SpinnerColumn,
            BarColumn,
            TextColumn,
            TimeElapsedColumn,
        )
        from rich.console import Group

        if not hasattr(_update_display, "live"):
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
            )
            _update_display.task = progress.add_task(
                "Detecting dead frames...", total=100
            )
            _update_display.progress = progress
            _update_display.live = Live(
                progress, console=err, refresh_per_second=4, transient=True
            )
            _update_display.live.start()

        s = monitor.stats
        _update_display.progress.update(
            _update_display.task, completed=s["pct"], description=s["msg"]
        )

        if monitor.enabled and hasattr(_update_display, "live"):
            lines = []
            gpu_parts = []
            if s.get("gpu_util") is not None:
                gpu_parts.append(f"GPU {s['gpu_util']:.0f}%")
            if s.get("gpu_temp") is not None:
                gpu_parts.append(f"{s['gpu_temp']:.0f}°C")
            if (
                s.get("vram_used_mb") is not None
                and s.get("vram_total_mb")
            ):
                gpu_parts.append(
                    f"VRAM {s['vram_used_mb']:.0f}/{s['vram_total_mb']:.0f} MB"
                )
            if gpu_parts:
                lines.append(
                    f"  {s.get('gpu_name', 'GPU')}: {' | '.join(gpu_parts)}"
                )

            cpu_parts = []
            if s.get("cpu_percent") is not None:
                cpu_parts.append(f"CPU {s['cpu_percent']:.0f}%")
            if s.get("ram_used_gb") is not None and s.get("ram_total_gb"):
                cpu_parts.append(
                    f"RAM {s['ram_used_gb']:.1f}/{s['ram_total_gb']:.1f} GB"
                )
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
                content.append(
                    Panel(
                        "\n".join(lines),
                        border_style="#22c55e",
                        padding=(0, 1),
                    )
                )
            _update_display.live.update(Group(*content))

    def _progress_cb(pct, msg):
        monitor.progress_callback(pct, msg)
        _update_display()

    try:
        result = run_deadframes(
            input_path=str(input.resolve()),
            output_path=str(output.resolve()),
            flow_threshold=0.5,
            motion_area_fraction=0.15,
            detect_scale=detect_scale,
            keep_talking=use_keep_talking,
            keep_camera=use_keep_camera,
            parallax_mode=parallax,
            auto=auto_mode,
            cadence=cadence,
            small_movements=small_movements,
            prores=prores,
            no_audio=no_audio,
            progress_cb=_progress_cb,
        )
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

    console.print()
    console.print(
        f"  Frames:  [accent]{result['kept_frames']}[/accent] kept / "
        f"[error]{result['dropped_frames']}[/error] dropped "
        f"([muted]{result['total_frames']} total[/])"
    )
    console.print(
        f"  Duration: [accent]{result['duration_after']:.2f}s[/accent]"
    )
    if result.get("flow_threshold") is not None:
        console.print(
            f"  Thresholds: flow=[accent]{result['flow_threshold']:.3f}[/accent] "
            f"area=[accent]{result['motion_area_fraction']:.4f}[/accent]"
        )
    ok(f"Saved: {output} ({monitor.stats['elapsed_s']:.1f}s)")
