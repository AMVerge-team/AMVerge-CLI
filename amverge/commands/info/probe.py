from __future__ import annotations

from pathlib import Path

import typer

from ...ui import banner, console, make_table, err


def _fmt_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h:
        return f"{h}h {m:02d}m {s:05.2f}s"
    if m:
        return f"{m}m {s:05.2f}s"
    return f"{s:.2f}s"


def probe(
    video: Path = typer.Argument(..., help="Video file to probe", exists=True),
    keyframes: bool = typer.Option(True, "--keyframes/--no-keyframes", help="Scan keyframe timestamps (slower)"),
    cache_dir: Path = typer.Option(None, "--cache-dir", help="Directory to check for scene cache"),
) -> None:
    """Show detailed V2 diagnostics: codec, HEVC, keyframes, scene cache."""
    from ...core.video.probe_utils import probe_video_fps, probe_video_duration, probe_video_dimensions
    from ...core.codec.codec_utils import check_if_hevc
    from ...core.infra.ipc import build_video_cache_prefix
    from ...ui import make_progress

    banner("probe")

    video = video.resolve()

    with make_progress(transient=True) as progress:
        task = progress.add_task("Reading video metadata...", total=None)

        fps = probe_video_fps(video)
        duration = probe_video_duration(video)
        width, height = probe_video_dimensions(video)
        total_frames = int(fps * duration)

        progress.update(task, description="Checking codec...")
        is_hevc = check_if_hevc(video)

        kf_times: list[float] = []
        if keyframes:
            progress.update(task, description="Scanning keyframes...")
            from ...core.keyframes.keyframe_align import get_keyframe_timestamps_pyav
            kf_times = get_keyframe_timestamps_pyav(str(video))

    console.print(f"[label]{video.name}[/]  [muted]{_fmt_duration(duration)}[/]\n")

    t = make_table(
        ("", "muted",  {"width": 14, "no_wrap": True}),
        ("", "label",  {}),
        title="Video",
    )
    t.add_row("Duration",   _fmt_duration(duration))
    t.add_row("Resolution", f"{width}x{height}")
    t.add_row("FPS",        f"{fps:.3f}")
    t.add_row("Codec",      "HEVC (H.265)" if is_hevc else "H.264 / other")
    t.add_row("Frames",     f"~{total_frames:,}")
    console.print(t)

    if keyframes:
        kf_count = len(kf_times)
        if kf_count >= 2:
            avg_gap = (kf_times[-1] - kf_times[0]) / (kf_count - 1)
            avg_gap_str = f"{avg_gap:.2f}s"
        elif kf_count == 1:
            avg_gap_str = "-"
        else:
            avg_gap_str = "-"

        t2 = make_table(
            ("", "muted",  {"width": 14, "no_wrap": True}),
            ("", "label",  {}),
            title="Keyframes",
        )
        t2.add_row("Count",    f"{kf_count:,}")
        t2.add_row("Avg. gap", avg_gap_str)
        if is_hevc:
            t2.add_row("Cut mode", "[warn]snapped_copy[/] (CPU) / [accent]reencode[/] (HEVC+CUDA)")
        else:
            t2.add_row("Cut mode", "[accent]smartcut[/] (H.264) or [accent]copy[/]")
        console.print(t2)

    search_dir = cache_dir.resolve() if cache_dir else video.parent
    cache_prefix = build_video_cache_prefix(video)
    secs_path = search_dir / f"{cache_prefix}_secs.npy"
    frames_path = search_dir / f"{cache_prefix}_frames.npy"
    cache_hit = secs_path.exists() and frames_path.exists()

    t3 = make_table(
        ("", "muted",  {"width": 14, "no_wrap": True}),
        ("", "label",  {}),
        title="Scene Cache",
    )
    t3.add_row("Prefix", cache_prefix)
    t3.add_row("Location", str(search_dir))
    if cache_hit:
        t3.add_row("Status", "[accent]cached[/] - detection will be skipped")
    else:
        t3.add_row("Status", "[muted]no cache - TransNetV2 will run on next detect[/]")
    console.print(t3)
