from __future__ import annotations

from pathlib import Path

import typer

from ...ui import banner, console, make_table, ok, fail, warn


def cache(
    directory: Path = typer.Argument(..., help="Directory to scan for scene caches", exists=True),
    video: Path = typer.Option(None, "--clear", metavar="VIDEO", help="Delete cache for this video"),
    clear_all: bool = typer.Option(False, "--clear-all", help="Delete all caches in directory"),
) -> None:
    """List or clear TransNetV2 scene caches (.npy files)."""
    from ...core.infra.ipc import build_video_cache_prefix

    banner("cache")

    directory = directory.resolve()

    if clear_all:
        files = list(directory.glob("scenes_*_secs.npy")) + list(directory.glob("scenes_*_frames.npy"))
        if not files:
            warn("No cache files found.")
            return
        for f in files:
            f.unlink()
        ok(f"Deleted {len(files)} cache files from {directory}")
        return

    if video is not None:
        video = video.resolve()
        if not video.exists():
            fail(f"Video not found: {video}")
            raise typer.Exit(1)
        prefix = build_video_cache_prefix(video)
        secs = directory / f"{prefix}_secs.npy"
        frames = directory / f"{prefix}_frames.npy"
        deleted = 0
        for f in (secs, frames):
            if f.exists():
                f.unlink()
                deleted += 1
        if deleted:
            ok(f"Cleared cache for {video.name} ({prefix})")
        else:
            warn(f"No cache found for {video.name} in {directory}")
        return

    secs_files = sorted(directory.glob("scenes_*_secs.npy"))
    if not secs_files:
        warn("No scene caches found.")
        return

    t = make_table(
        ("Prefix",   "accent", {"no_wrap": True}),
        ("Secs",     "label",  {"justify": "right"}),
        ("Frames",   "muted",  {"justify": "right"}),
        ("Size",     "muted",  {"justify": "right"}),
        title=f"Scene Caches in {directory}",
    )

    import numpy as np

    total_bytes = 0
    for secs_path in secs_files:
        prefix = secs_path.stem.replace("_secs", "")
        frames_path = directory / f"{prefix}_frames.npy"

        try:
            secs_arr = np.load(secs_path)
            n_scenes = len(secs_arr)
        except Exception:
            n_scenes = -1

        has_frames = frames_path.exists()
        size = secs_path.stat().st_size + (frames_path.stat().st_size if has_frames else 0)
        total_bytes += size
        size_str = f"{size / 1024:.1f} KB"

        frames_label = "[accent]yes[/]" if has_frames else "[error]missing[/]"
        t.add_row(prefix, str(n_scenes) if n_scenes >= 0 else "?", frames_label, size_str)

    console.print(t)
    console.print(f"[muted]Total: {len(secs_files)} cache(s), {total_bytes / 1024:.1f} KB[/]")
