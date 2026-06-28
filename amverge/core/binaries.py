import sys
from pathlib import Path
from shutil import which


def _find(name: str) -> str:
    """Find ffmpeg/ffprobe: PATH first, then common local layouts."""
    exe = name + (".exe" if sys.platform == "win32" else "")

    found = which(exe) or which(name)
    if found:
        p = Path(found)
        if p.is_absolute() and p.exists():
            return str(p)

    cwd = Path.cwd()
    candidates = [
        cwd / exe,
        cwd / name,
        cwd / "bin" / exe,
        cwd / "bin" / name,
    ]

    for c in candidates:
        if c.exists():
            return str(c)

    return exe  # let subprocess raise a clear error


def get_ffmpeg() -> str:
    return _find("ffmpeg")


def get_ffprobe() -> str:
    return _find("ffprobe")
