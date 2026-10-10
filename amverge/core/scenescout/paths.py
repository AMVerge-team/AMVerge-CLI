"""Where Scene Scout databases live.

One rule decides everything here: **the app owns the location, the CLI is told
about it.** The desktop app keeps every kind of storage under one user-chosen
root and moves the whole tree when that root changes, so the CLI must never
compute a path of its own when the app has given it one.

Resolution order, highest first:

1. ``--root`` on the command line
2. ``AMVERGE_SCENE_SCOUT_DIR`` in the environment
3. the standalone default under the user's data directory

The app always passes (1), so its databases follow the storage location the user
picked in Settings. A developer running the CLI on its own gets (3) and never has
to configure anything.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# The folder name is shared with the app's Rust side
# (`resolve_scene_scout_storage_dir`). Changing it here orphans every database
# the app already wrote, so change both together.
STORAGE_DIR_NAME = "amverge-scene-scout"

DB_SUFFIX = ".scoutdb"

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def default_root() -> Path:
    """Standalone location, used when neither the app nor the user says otherwise."""
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif os.uname().sysname == "Darwin":  # type: ignore[attr-defined]
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "AMVerge" / STORAGE_DIR_NAME


def resolve_root(explicit: str | Path | None = None) -> Path:
    """The directory holding every Scene Scout database."""
    if explicit:
        return Path(explicit).expanduser().resolve()

    from_env = os.environ.get("AMVERGE_SCENE_SCOUT_DIR", "").strip()
    if from_env:
        return Path(from_env).expanduser().resolve()

    return default_root()


def sanitize_db_name(name: str) -> str:
    """Turn a user-typed database name into a safe file stem.

    The name reaches this from a text field in the app, so it is untrusted: path
    separators and ``..`` here would let a database be written outside the
    storage root.
    """
    cleaned = _UNSAFE.sub("_", name.strip()).strip("._-")
    return cleaned[:64] or "database"


SUPPORTED_DB_SUFFIXES = {".scoutdb", ".db", ".scdb", ".sqlite", ".sqlite3"}


def looks_like_path(value: str | Path) -> bool:
    s = str(value)
    return any(s.lower().endswith(suf) for suf in SUPPORTED_DB_SUFFIXES) or "/" in s or "\\" in s


def db_path(name: str | Path, root: str | Path | None = None) -> Path:
    s = str(name)
    if looks_like_path(s):
        path = Path(s).expanduser()
        if path.is_file():
            return path.resolve()
        if path.suffix.lower() not in SUPPORTED_DB_SUFFIXES:
            path = path.with_suffix(DB_SUFFIX)
        return path.resolve()
    return resolve_root(root) / f"{sanitize_db_name(s)}{DB_SUFFIX}"


def resolve_db_path(database: str | Path, root: str | Path | None = None) -> Path:
    p = Path(database)
    if p.is_file():
        return p.resolve()
    return db_path(database, root)


def list_db_paths(root: str | Path | None = None) -> list[Path]:
    directory = resolve_root(root)
    if not directory.is_dir():
        return []
    found: list[Path] = []
    for suffix in (".scoutdb", ".db", ".scdb"):
        found.extend(directory.glob(f"*{suffix}"))
    seen = set()
    result = []
    for p in sorted(found):
        resolved = p.resolve()
        if p.is_file() and resolved not in seen:
            seen.add(resolved)
            result.append(p)
    return result


def ensure_root(root: str | Path | None = None) -> Path:
    directory = resolve_root(root)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
