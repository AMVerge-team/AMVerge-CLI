from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

import typer

from ..core.binaries import get_ffmpeg
from ..ui import banner, ok, fail

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def merge(
    clips: List[Path] = typer.Argument(..., help="Clip files to merge in order"),
    output: Path = typer.Option(..., "--output", "-o", help="Output file"),
) -> None:
    """Merge multiple clips into one file via FFmpeg concat."""
    for clip in clips:
        if not clip.exists():
            fail(f"Not found: {clip}")
            raise typer.Exit(1)

    banner("merge")

    ff = get_ffmpeg()
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        concat_file = f.name
        for clip in clips:
            f.write(f"file '{str(clip.resolve()).replace(chr(92), '/')}'\n")

    try:
        subprocess.run(
            [ff, "-y", "-f", "concat", "-safe", "0", "-i", concat_file, "-c", "copy", str(output)],
            capture_output=True,
            creationflags=CREATE_NO_WINDOW,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        fail(f"ffmpeg failed: {e.stderr.decode(errors='replace')[-500:]}")
        raise typer.Exit(1)
    finally:
        os.unlink(concat_file)

    ok(f"{len(clips)} clips → {output}")
