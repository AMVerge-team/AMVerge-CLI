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


def looks_like_path(value: str) -> bool:
    """True when `value` names a file rather than a database inside the root.

    The app lets a user save a database anywhere they like, so `--db` has to
    accept both "My Series" and "D:/somewhere/My Series.scoutdb". A separator or
    the suffix is enough to tell them apart; a bare name never has either.
    """
    return value.endswith(DB_SUFFIX) or "/" in value or "\\" in value


def db_path(name: str, root: str | Path | None = None) -> Path:
    """Absolute path of a database.

    Accepts either a name to resolve inside the root, or a path to a database
    the user put somewhere of their own choosing.
    """
    if looks_like_path(name):
        path = Path(name).expanduser()
        if path.suffix != DB_SUFFIX:
            path = path.with_suffix(DB_SUFFIX)
        return path.resolve()
    return resolve_root(root) / f"{sanitize_db_name(name)}{DB_SUFFIX}"


def list_db_paths(root: str | Path | None = None) -> list[Path]:
    """Every database in the root, oldest name first. Missing root is not an error."""
    directory = resolve_root(root)
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.glob(f"*{DB_SUFFIX}") if p.is_file())


def ensure_root(root: str | Path | None = None) -> Path:
    directory = resolve_root(root)
    directory.mkdir(parents=True, exist_ok=True)
    return directory
