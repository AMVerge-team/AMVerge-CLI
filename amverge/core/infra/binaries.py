from __future__ import annotations

"""FFmpeg/ffprobe binary resolution.

Searches for ffmpeg and ffprobe executables in frozen build directories
(PyInstaller _internal), the project root, ``bin/`` subdirectories, and
finally the system PATH. If nothing is found, returns a path that will
produce a clear error when used.

Example:
    >>> from amverge.core.infra.binaries import get_ffmpeg, get_ffprobe
    >>> ffmpeg = get_ffmpeg()
    >>> ffprobe = get_ffprobe()
"""

import os
import sys
from pathlib import Path
from shutil import which

if getattr(sys, "frozen", False):
    _ROOT = Path(sys.executable).resolve().parent
else:
    _ROOT = Path(__file__).resolve().parent.parent


def _platform_names(name: str) -> list[str]:
    if os.name == "nt" and not name.lower().endswith(".exe"):
        return [f"{name}.exe", name]
    return [name]


def get_binary(name: str) -> str:
    """Resolve a binary executable by name.

    Searches: frozen bundle (_internal/), project root, bin/, then PATH.
    Returns the first match or a best-guess path (caller should verify).

    Args:
        name: Executable name (e.g. ``"ffmpeg"``, ``"ffprobe"``).

    Returns:
        Absolute path to the binary if found, or the first search path
        joined with the name as a fallback.
    """
    search_dirs = [
        _ROOT / "_internal",
        _ROOT,
        _ROOT / "bin",
    ]

    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        search_dirs.extend([
            exe_dir / "_internal",
            exe_dir,
            exe_dir / "bin",
        ])

    try:
        from .ffmpeg_bootstrap import get_portable_ffmpeg_path as _pfp
        pp = _pfp()
        if pp:
            search_dirs.append(Path(pp).parent)
    except Exception:
        pass

    try:
        from .ffmpeg_bootstrap import get_portable_ffprobe_path as _pfpp
        pp = _pfpp()
        if pp and Path(pp).parent not in search_dirs:
            search_dirs.append(Path(pp).parent)
    except Exception:
        pass

    names = _platform_names(name)

    for directory in search_dirs:
        for candidate_name in names:
            candidate = directory / candidate_name
            if candidate.exists():
                return str(candidate)

    for candidate_name in names:
        found = which(candidate_name)
        if found:
            found_path = Path(found)
            if found_path.is_absolute() and found_path.exists():
                return str(found_path)

    return str(search_dirs[0] / names[0])


def get_ffmpeg() -> str:
    """Resolve the ffmpeg binary path. See :func:`get_binary`."""
    return get_binary("ffmpeg")


def get_ffprobe() -> str:
    """Resolve the ffprobe binary path. See :func:`get_binary`."""
    return get_binary("ffprobe")
