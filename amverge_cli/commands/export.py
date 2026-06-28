from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import typer

from ..core.binaries import get_ffmpeg
from ..ui import banner, console, err, make_progress, make_table, ok, fail, dim

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
VALID_CODECS = {"copy", "h264", "hevc", "h265"}


def _parse_select(select: str, max_index: int) -> list[int]:
    indices: set[int] = set()
    for part in select.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            indices.update(range(int(lo), int(hi) + 1))
        else:
            indices.add(int(part))
    return sorted(i for i in indices if 0 <= i <= max_index)


def export(
    video: Path = typer.Argument(..., help="Source video file", exists=True),
    scenes: Path = typer.Option(..., "--scenes", "-s", help="scenes.json from detect", exists=True),
    output: Path = typer.Option(Path("export"), "--output", "-o", help="Output directory"),
    select: Optional[str] = typer.Option(None, "--select", help='Indices: "0,2,5-8" (default: all)'),
    merge: bool = typer.Option(False, "--merge", help="Merge selected clips into one file"),
    codec: str = typer.Option("copy", "--codec", help="copy · h264 · hevc"),
) -> None:
    """Export selected scenes from a detect run."""
    if codec not in VALID_CODECS:
        fail(f"Unknown codec '{codec}'. Valid: {', '.join(sorted(VALID_CODECS))}")
        raise typer.Exit(1)
    if codec == "h265":
        codec = "hevc"

    banner("export")

    payload = json.loads(scenes.read_text())
    all_scenes: list[dict] = payload.get("scenes", payload) if isinstance(payload, dict) else payload

    if not all_scenes:
        fail("No scenes in JSON.")
        raise typer.Exit(1)

    max_idx = max(s["scene_index"] for s in all_scenes)
    selected = (
        [s for s in all_scenes if s["scene_index"] in _parse_select(select, max_idx)]
        if select else all_scenes
    )

    if not selected:
        fail("No scenes matched selection.")
        raise typer.Exit(1)

    output.mkdir(parents=True, exist_ok=True)
    ff = get_ffmpeg()

    if merge:
        _export_merged(selected, output, ff, codec)
    else:
        _export_individual(selected, output, ff, codec)


def _copy(src: str, dst: str, ff: str) -> None:
    subprocess.run(
        [ff, "-y", "-i", src, "-c", "copy", dst],
        capture_output=True, creationflags=CREATE_NO_WINDOW, check=True,
    )


def _encode(src: str, dst: str, ff: str, codec: str) -> None:
    subprocess.run(
        [ff, "-y", "-i", src, "-c:v", codec, "-c:a", "aac", "-b:a", "160k", dst],
        capture_output=True, creationflags=CREATE_NO_WINDOW, check=True,
    )


def _export_individual(scenes: list[dict], output: Path, ff: str, codec: str) -> None:
    with make_progress(show_count=True) as progress:
        task = progress.add_task(f"Exporting {len(scenes)} clips", total=len(scenes))
        for s in scenes:
            idx = s["scene_index"]
            dst = str(output / f"scene_{idx:04d}.mp4")
            if codec == "copy":
                _copy(s["path"], dst, ff)
            else:
                _encode(s["path"], dst, ff, codec)
            progress.advance(task)
            progress.update(task, description=f"Exported scene_{idx:04d}")

    ok(f"{len(scenes)} clips → {output}")


def _export_merged(scenes: list[dict], output: Path, ff: str, codec: str) -> None:
    with make_progress() as progress:
        task = progress.add_task(f"Merging {len(scenes)} clips", total=1)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            concat_file = f.name
            for s in scenes:
                f.write(f"file '{s['path'].replace(chr(92), '/')}'\n")

        dst = str(output / "merged.mp4")
        try:
            cmd = [ff, "-y", "-f", "concat", "-safe", "0", "-i", concat_file]
            cmd += ["-c", "copy"] if codec == "copy" else ["-c:v", codec, "-c:a", "aac"]
            cmd.append(dst)
            subprocess.run(cmd, capture_output=True, creationflags=CREATE_NO_WINDOW, check=True)
        finally:
            os.unlink(concat_file)

        progress.update(task, completed=1)

    ok(f"Merged → {dst}")
