"""SQLite storage for Scene Scout (schema v4).

The schema is shared with the standalone Scene Scout project
(https://github.com/Mark-Shun/scene-scout) so databases produced by either tool
open cleanly in the other. Keep core tables aligned across both repositories.

Only stdlib imports at module scope (numpy imported lazily where needed for array
packing). Listing databases and reporting status must work without torch installed.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Optional

from .cache import invalidate_embedding_cache
from .fingerprint import verify_content_fingerprint
from .paths import db_path, list_db_paths, resolve_db_path
from .types import (
    DatabaseInfo,
    IndexedVideo,
    EMBEDDING_MODEL_VERSION,
    SCHEMA_VERSION,
    SchemaUpgradeRequiredError,
    UnsupportedSchemaError,
)

PAGE_SIZE = 16384
MMAP_SIZE = 268_435_456  # 256 MiB

SCHEMA = """
CREATE TABLE IF NOT EXISTS db_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS media_roots (
    root_path TEXT PRIMARY KEY,
    added_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS preprocessing_configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT NOT NULL,
    detector TEXT NOT NULL DEFAULT 'unknown',
    sampling_strategy TEXT NOT NULL DEFAULT 'unknown',
    sampling_config TEXT NOT NULL DEFAULT '{}',
    UNIQUE(model_id, detector, sampling_strategy, sampling_config)
);

CREATE TABLE IF NOT EXISTS processed_videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filepath TEXT UNIQUE NOT NULL,
    modified_at REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'completed',
    scene_count INTEGER NOT NULL DEFAULT 0,
    config_id INTEGER REFERENCES preprocessing_configs(id),
    file_size INTEGER,
    content_fingerprint TEXT,
    duration_ms INTEGER,
    fps REAL,
    width INTEGER,
    height INTEGER
);

CREATE TABLE IF NOT EXISTS video_embeddings (
    video_id INTEGER PRIMARY KEY,
    scene_count INTEGER NOT NULL,
    dim INTEGER NOT NULL,
    dtype TEXT NOT NULL DEFAULT 'float32',
    embeddings BLOB NOT NULL,
    FOREIGN KEY (video_id) REFERENCES processed_videos(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS scenes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL,
    scene_index INTEGER NOT NULL,
    start_time_ms INTEGER NOT NULL,
    end_time_ms INTEGER NOT NULL,
    FOREIGN KEY (video_id) REFERENCES processed_videos(id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_scene_video_scene ON scenes(video_id, scene_index);

CREATE TABLE IF NOT EXISTS scene_thumbnails (
    scene_id INTEGER PRIMARY KEY,
    thumbnail BLOB NOT NULL,
    FOREIGN KEY (scene_id) REFERENCES scenes(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS index_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    is_directory BOOLEAN NOT NULL DEFAULT 0,
    recursive BOOLEAN NOT NULL DEFAULT 1,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# AMVerge-only sidecar table ignored by standalone Scene Scout; raw numbers, tags derived on read
MOTION_SCHEMA = """
CREATE TABLE IF NOT EXISTS scene_motion (
    scene_id INTEGER PRIMARY KEY,
    motion_version INTEGER NOT NULL,
    start_raw TEXT,
    end_raw TEXT,
    FOREIGN KEY (scene_id) REFERENCES scenes(id) ON DELETE CASCADE
);
"""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _file_time_iso(path: Path) -> str:
    try:
        st = path.stat()
        ts = getattr(st, "st_birthtime", None) or st.st_ctime or st.st_mtime
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    except OSError:
        return _utc_now_iso()


def resolve_caller_identity() -> tuple[str, str]:
    """Return ``(app_name, app_version)`` for metadata provenance."""
    from ...__version__ import __version__

    caller_app = os.environ.get("AMVERGE_CALLER_APP", "").strip()
    caller_ver = os.environ.get("AMVERGE_CALLER_VERSION", "").strip()
    if caller_app:
        return caller_app, caller_ver or "unknown"
    return "amverge-cli", __version__


@contextmanager
def connect(path: str | Path, timeout: float = 10.0) -> Iterator[sqlite3.Connection]:
    """Open a database with the pragmas indexing throughput depends on."""
    conn = sqlite3.connect(str(path), timeout=timeout)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        yield conn
        conn.commit()
    finally:
        conn.close()


@contextmanager
def connect_ro(path: str | Path, timeout: float = 10.0) -> Iterator[sqlite3.Connection]:
    """Open a database for read-only queries with memory-mapped I/O enabled."""
    conn = sqlite3.connect(str(path), timeout=timeout)
    try:
        conn.execute(f"PRAGMA mmap_size={MMAP_SIZE}")
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Metadata, Preprocessing Configs & Media Roots Helpers
# ---------------------------------------------------------------------------


def _seed_or_touch_metadata(
    conn: sqlite3.Connection,
    *,
    is_fresh: bool,
    fallback_created_at: Optional[str] = None,
) -> None:
    app_name, app_ver = resolve_caller_identity()
    now = _utc_now_iso()
    existing = dict(conn.execute("SELECT key, value FROM db_metadata").fetchall())

    if "database_id" not in existing:
        conn.execute(
            "INSERT OR REPLACE INTO db_metadata (key, value) VALUES ('database_id', ?)",
            (str(uuid.uuid4()),),
        )

    if "created_by_app" not in existing:
        conn.execute(
            "INSERT OR REPLACE INTO db_metadata (key, value) VALUES ('created_by_app', ?)",
            (app_name if is_fresh else "unknown",),
        )
    if "created_by_version" not in existing:
        conn.execute(
            "INSERT OR REPLACE INTO db_metadata (key, value) VALUES ('created_by_version', ?)",
            (app_ver if is_fresh else "unknown",),
        )
    if "created_at" not in existing:
        conn.execute(
            "INSERT OR REPLACE INTO db_metadata (key, value) VALUES ('created_at', ?)",
            (now if is_fresh else (fallback_created_at or now),),
        )

    for k, v in (
        ("schema_version", str(SCHEMA_VERSION)),
        ("updated_by_app", app_name),
        ("updated_by_version", app_ver),
        ("updated_at", now),
    ):
        conn.execute("INSERT OR REPLACE INTO db_metadata (key, value) VALUES (?, ?)", (k, v))


def _touch_updated_metadata(conn: sqlite3.Connection) -> None:
    app_name, app_ver = resolve_caller_identity()
    now = _utc_now_iso()
    for k, v in (
        ("schema_version", str(SCHEMA_VERSION)),
        ("updated_by_app", app_name),
        ("updated_by_version", app_ver),
        ("updated_at", now),
    ):
        conn.execute("INSERT OR REPLACE INTO db_metadata (key, value) VALUES (?, ?)", (k, v))


def get_or_create_config(
    conn: sqlite3.Connection,
    model_id: str = EMBEDDING_MODEL_VERSION,
    detector: str = "unknown",
    sampling_strategy: str = "unknown",
    sampling_config: Optional[dict[str, Any] | str] = None,
) -> int:
    model_id = (model_id or EMBEDDING_MODEL_VERSION).strip() or EMBEDDING_MODEL_VERSION
    detector = (detector or "unknown").strip() or "unknown"
    sampling_strategy = (sampling_strategy or "unknown").strip() or "unknown"
    if sampling_config is None:
        cfg_str = "{}"
    elif isinstance(sampling_config, str):
        cfg_str = sampling_config.strip() or "{}"
    else:
        cfg_str = json.dumps(sampling_config, sort_keys=True, separators=(",", ":"))

    conn.execute(
        """
        INSERT OR IGNORE INTO preprocessing_configs
            (model_id, detector, sampling_strategy, sampling_config)
        VALUES (?, ?, ?, ?)
        """,
        (model_id, detector, sampling_strategy, cfg_str),
    )
    row = conn.execute(
        """
        SELECT id FROM preprocessing_configs
        WHERE model_id = ? AND detector = ? AND sampling_strategy = ? AND sampling_config = ?
        """,
        (model_id, detector, sampling_strategy, cfg_str),
    ).fetchone()
    return int(row[0])


_SHALLOW_POSIX_ROOTS = {"", "/", "/users", "/home", "/volumes", "/mnt", "/media", "/var", "/tmp"}


def _norm_slash_path(raw: str) -> str:
    s = str(raw).replace("\\", "/").strip()
    while "//" in s and not s.startswith("//"):
        s = s.replace("//", "/")
    if len(s) > 1 and s.endswith("/"):
        s = s.rstrip("/")
    return s


def _is_usable_media_root(norm_dir: str) -> bool:
    """Reject bare drive letters or top-level OS user containers as media roots."""
    if not norm_dir:
        return False
    low = norm_dir.lower()
    if low in _SHALLOW_POSIX_ROOTS:
        return False
    # Windows drive root like "C:" or "C:/"
    if len(low) <= 3 and len(low) >= 2 and low[1] == ":":
        return False
    if low in ("c:/users",):
        return False
    parts = [p for p in norm_dir.split("/") if p]
    if low.startswith("/users/") or low.startswith("/home/") or (len(parts) >= 2 and parts[0].endswith(":") and parts[1].lower() == "users"):
        # Require at least <drive_or_root>/Users/<username>/<folder>
        expected = 4 if parts[0].endswith(":") else 3
        return len(parts) >= expected
    if low.startswith("/volumes/") or low.startswith("/mnt/") or low.startswith("/media/"):
        return len(parts) >= 3
    return len(parts) >= 2


def _common_dir_prefix(dir_a: str, dir_b: str) -> Optional[str]:
    pa = dir_a.split("/")
    pb = dir_b.split("/")
    common: list[str] = []
    for a, b in zip(pa, pb):
        if a.lower() == b.lower():
            common.append(a)
        else:
            break
    if not common:
        return None
    joined = "/".join(common)
    if dir_a.startswith("/") and not joined.startswith("/"):
        joined = "/" + joined
    return joined


def _path_is_under_root(norm_filepath: str, norm_root: str) -> bool:
    fp_low = norm_filepath.lower()
    rt_low = norm_root.lower().rstrip("/")
    return fp_low.startswith(rt_low + "/")


def record_media_root(conn: sqlite3.Connection, filepath: str, explicit_root: Optional[str] = None) -> None:
    """Record or consolidate a common root directory hint for ``filepath``."""
    now = _utc_now_iso()
    if explicit_root:
        cand = _norm_slash_path(explicit_root)
        if _is_usable_media_root(cand):
            conn.execute(
                "INSERT OR IGNORE INTO media_roots (root_path, added_at) VALUES (?, ?)",
                (cand, now),
            )
            return

    norm_fp = _norm_slash_path(filepath)
    if "/" not in norm_fp:
        return
    parent = str(PurePosixPath(norm_fp).parent)
    if not _is_usable_media_root(parent):
        return

    existing = [r[0] for r in conn.execute("SELECT root_path FROM media_roots").fetchall()]
    for root in existing:
        if _path_is_under_root(norm_fp, root):
            return
        if _path_is_under_root(root + "/x", parent) and root.lower() != parent.lower():
            # New file sits in an ancestor directory of an existing narrower root
            conn.execute("DELETE FROM media_roots WHERE root_path = ?", (root,))
            conn.execute("INSERT OR IGNORE INTO media_roots (root_path, added_at) VALUES (?, ?)", (parent, now))
            return
        common = _common_dir_prefix(root, parent)
        if common and _is_usable_media_root(common):
            conn.execute("DELETE FROM media_roots WHERE root_path = ?", (root,))
            conn.execute("INSERT OR IGNORE INTO media_roots (root_path, added_at) VALUES (?, ?)", (common, now))
            return

    conn.execute("INSERT OR IGNORE INTO media_roots (root_path, added_at) VALUES (?, ?)", (parent, now))


# ---------------------------------------------------------------------------
# Database Lifecycle & Schema v4 Migration
# ---------------------------------------------------------------------------


def create_database(name: str, root: str | Path | None = None) -> Path:
    path = db_path(name, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.stat().st_size > 0:
        ensure_v4_ready(path)
        return path

    conn = sqlite3.connect(str(path), timeout=10.0, isolation_level=None)
    try:
        conn.execute(f"PRAGMA page_size = {PAGE_SIZE}")
        conn.execute("PRAGMA auto_vacuum = INCREMENTAL")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA)
        conn.executescript(MOTION_SCHEMA)
        conn.execute("BEGIN")
        _seed_or_touch_metadata(conn, is_fresh=True)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        conn.execute("COMMIT")
    finally:
        conn.close()

    invalidate_embedding_cache(path)
    return path


def has_motion_table(conn: sqlite3.Connection) -> bool:
    return (
        conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='scene_motion'").fetchone()
        is not None
    )


def _table_names(conn: sqlite3.Connection) -> set[str]:
    return {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}


def _column_names(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def is_valid_database(path: str | Path) -> bool:
    p = Path(path)
    if not p.is_file():
        return False
    try:
        with connect_ro(p) as conn:
            tables = _table_names(conn)
            return bool(tables.intersection({"processed_videos", "scenes", "scene_embeddings", "image_embeddings"}))
    except Exception:
        return False


def _open_strict_ro(path: Path) -> sqlite3.Connection:
    """Open strictly read-only without creating WAL/SHM sidecars."""
    try:
        return sqlite3.connect(f"file:{path.resolve().as_posix()}?mode=ro", uri=True, timeout=10.0)
    except sqlite3.OperationalError:
        return sqlite3.connect(str(path), timeout=10.0)


def inspect_database_schema(path: str | Path) -> dict[str, Any]:
    """Inspect schema version and metadata strictly read-only without modifying the file."""
    p = Path(path)
    if not p.is_file():
        return {
            "status": "missing",
            "schema_version": 0,
            "needs_self_heal": False,
            "meta": {},
            "roots": [],
            "video_count": 0,
            "scene_count": 0,
            "model_version": EMBEDDING_MODEL_VERSION,
        }

    meta: dict[str, str] = {}
    roots: list[str] = []
    videos = scenes = 0
    model_version = EMBEDDING_MODEL_VERSION
    uv = 0

    try:
        conn = _open_strict_ro(p)
    except sqlite3.DatabaseError:
        return {
            "status": "invalid",
            "schema_version": 0,
            "needs_self_heal": False,
            "meta": {},
            "roots": [],
            "video_count": 0,
            "scene_count": 0,
            "model_version": EMBEDDING_MODEL_VERSION,
        }

    try:
        tables = _table_names(conn)
        has_core = bool(tables.intersection({"processed_videos", "scenes", "scene_embeddings", "image_embeddings"}))
        if not has_core:
            return {
                "status": "invalid",
                "schema_version": 0,
                "needs_self_heal": False,
                "meta": {},
                "roots": [],
                "video_count": 0,
                "scene_count": 0,
                "model_version": EMBEDDING_MODEL_VERSION,
            }

        uv = int(conn.execute("PRAGMA user_version").fetchone()[0])
        if "db_metadata" in tables:
            meta = dict(conn.execute("SELECT key, value FROM db_metadata").fetchall())
        if "media_roots" in tables:
            roots = [r[0] for r in conn.execute("SELECT root_path FROM media_roots ORDER BY root_path").fetchall()]

        if "processed_videos" in tables:
            pv_cols = _column_names(conn, "processed_videos")
            if "scene_count" in pv_cols:
                row = conn.execute(
                    "SELECT COUNT(*), COALESCE(SUM(scene_count), 0) FROM processed_videos"
                ).fetchone()
                videos, scenes = int(row[0]), int(row[1])
            else:
                videos = int(conn.execute("SELECT COUNT(*) FROM processed_videos").fetchone()[0])
                if "scenes" in tables:
                    scenes = int(conn.execute("SELECT COUNT(*) FROM scenes").fetchone()[0])
                elif "scene_embeddings" in tables:
                    scenes = int(conn.execute("SELECT COUNT(*) FROM scene_embeddings").fetchone()[0])

            if "preprocessing_configs" in tables and "config_id" in pv_cols:
                mrow = conn.execute(
                    """
                    SELECT pc.model_id
                    FROM processed_videos pv
                    JOIN preprocessing_configs pc ON pc.id = pv.config_id
                    LIMIT 1
                    """
                ).fetchone()
                if mrow and mrow[0]:
                    model_version = str(mrow[0])
                else:
                    crow = conn.execute("SELECT model_id FROM preprocessing_configs LIMIT 1").fetchone()
                    if crow and crow[0]:
                        model_version = str(crow[0])
            elif "model_version" in pv_cols:
                mrow = conn.execute("SELECT model_version FROM processed_videos LIMIT 1").fetchone()
                if mrow and mrow[0]:
                    model_version = str(mrow[0])

        if uv > SCHEMA_VERSION:
            return {
                "status": "unsupported_newer",
                "schema_version": uv,
                "needs_self_heal": False,
                "meta": meta,
                "roots": roots,
                "video_count": videos,
                "scene_count": scenes,
                "model_version": model_version,
            }

        has_v4_core = {
            "processed_videos",
            "scenes",
            "video_embeddings",
            "db_metadata",
            "preprocessing_configs",
        }.issubset(tables)
        if has_v4_core:
            needs_heal = (
                uv != SCHEMA_VERSION
                or "scene_embeddings" in tables
                or "image_embeddings" in tables
                or "scene_thumbnails" not in tables
                or "media_roots" not in tables
            )
            return {
                "status": "ready",
                "schema_version": SCHEMA_VERSION,
                "needs_self_heal": needs_heal,
                "meta": meta,
                "roots": roots,
                "video_count": videos,
                "scene_count": scenes,
                "model_version": model_version,
            }

        inferred_ver = uv
        if inferred_ver <= 0:
            if "scene_embeddings" in tables and "video_id" in _column_names(conn, "scene_embeddings"):
                inferred_ver = 3
            elif "index_queue" in tables:
                inferred_ver = 2
            else:
                inferred_ver = 1

        return {
            "status": "needs_upgrade",
            "schema_version": inferred_ver,
            "needs_self_heal": False,
            "meta": meta,
            "roots": roots,
            "video_count": videos,
            "scene_count": scenes,
            "model_version": model_version,
        }
    except sqlite3.DatabaseError:
        return {
            "status": "invalid",
            "schema_version": uv,
            "needs_self_heal": False,
            "meta": meta,
            "roots": roots,
            "video_count": videos,
            "scene_count": scenes,
            "model_version": model_version,
        }
    finally:
        conn.close()


def ensure_v4_ready(path: str | Path) -> bool:
    """Verify ``path`` is a ready v4 DB (self-healing if polluted by an old v3 app).

    Raises ``SchemaUpgradeRequiredError`` for pre-v4 databases instead of
    silently mutating them, and ``UnsupportedSchemaError`` for ``v5+`` databases.
    """
    p = Path(path).resolve()
    info = inspect_database_schema(p)
    status = info["status"]
    if status == "missing":
        raise FileNotFoundError(f"Database file not found: {p}")
    if status == "invalid":
        raise ValueError(f"'{p.name}' is not a valid Scene Scout database (missing core tables).")
    if status == "unsupported_newer":
        meta = info["meta"]
        raise UnsupportedSchemaError(
            p,
            int(info["schema_version"]),
            updated_by_app=meta.get("updated_by_app") or meta.get("created_by_app"),
            updated_by_version=meta.get("updated_by_version") or meta.get("created_by_version"),
        )
    if status == "needs_upgrade":
        raise SchemaUpgradeRequiredError(p, int(info["schema_version"]))
    if info["needs_self_heal"]:
        return migrate_database(p)
    return True


def create_schema_backup(path: str | Path, schema_version: int = 3) -> Path:
    """Create a consistent SQLite backup at ``<filename>.v<version>.bak`` without clobbering existing backups."""
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Database file not found: {p}")

    db_size = p.stat().st_size
    try:
        free_bytes = shutil.disk_usage(p.parent).free
        if free_bytes < db_size + 4 * 1024 * 1024:
            raise OSError(f"Insufficient disk space in {p.parent} to back up {p.name}.")
    except FileNotFoundError:
        pass

    backup_path = p.with_name(f"{p.name}.v{schema_version}.bak")
    if backup_path.exists():
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = p.with_name(f"{p.name}.v{schema_version}.{ts}.bak")

    src = sqlite3.connect(str(p), timeout=30.0)
    dst = sqlite3.connect(str(backup_path), timeout=30.0)
    try:
        src.backup(dst)
        dst.commit()
    finally:
        dst.close()
        src.close()
    return backup_path


def create_v3_backup(path: str | Path) -> Path:
    """Backward-compatible wrapper for creating a ``.v3.bak`` SQLite backup."""
    return create_schema_backup(path, schema_version=3)


def upgrade_database(path: str | Path, *, backup: bool = True) -> tuple[DatabaseInfo, Optional[Path]]:
    """Explicitly upgrade a pre-v4 database to schema v4 (optionally saving a ``.v<N>.bak``)."""
    p = Path(path).resolve()
    info = inspect_database_schema(p)
    status = info["status"]
    if status == "missing":
        raise FileNotFoundError(f"Database file not found: {p}")
    if status == "invalid":
        raise ValueError(f"'{p.name}' is not a valid Scene Scout database (missing core tables).")
    if status == "unsupported_newer":
        meta = info["meta"]
        raise UnsupportedSchemaError(
            p,
            int(info["schema_version"]),
            updated_by_app=meta.get("updated_by_app") or meta.get("created_by_app"),
            updated_by_version=meta.get("updated_by_version") or meta.get("created_by_version"),
        )

    backup_path: Optional[Path] = None
    if status == "needs_upgrade":
        if backup:
            backup_path = create_schema_backup(p, int(info.get("schema_version") or 3))
        migrate_database(p)
    elif info["needs_self_heal"]:
        migrate_database(p)

    return database_info(p), backup_path


def _rebuild_scene_motion_fk(conn: sqlite3.Connection) -> None:
    """Ensure ``scene_motion`` references ``scenes(id)`` rather than dropped ``scene_embeddings(id)``."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='scene_motion'"
    ).fetchone()
    if row is None:
        conn.execute(MOTION_SCHEMA.strip())
        return
    sql_def = (row[0] or "").lower()
    if "scene_embeddings" not in sql_def:
        return
    conn.execute(
        """
        CREATE TABLE scene_motion_new (
            scene_id INTEGER PRIMARY KEY,
            motion_version INTEGER NOT NULL,
            start_raw TEXT,
            end_raw TEXT,
            FOREIGN KEY (scene_id) REFERENCES scenes(id) ON DELETE CASCADE
        )
        """
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO scene_motion_new (scene_id, motion_version, start_raw, end_raw)
        SELECT m.scene_id, m.motion_version, m.start_raw, m.end_raw
        FROM scene_motion m
        WHERE m.scene_id IN (SELECT id FROM scenes)
        """
    )
    conn.execute("DROP TABLE scene_motion")
    conn.execute("ALTER TABLE scene_motion_new RENAME TO scene_motion")


def _compact_database_v4(path: Path) -> None:
    """Apply 16 KB page_size + INCREMENTAL auto_vacuum and reclaim dropped legacy table space."""
    try:
        db_size = path.stat().st_size
        free_bytes = shutil.disk_usage(path.parent).free
        if free_bytes < int(db_size * 1.15) + 4 * 1024 * 1024:
            return
    except OSError:
        return

    conn = sqlite3.connect(str(path), timeout=10.0, isolation_level=None)
    switched_to_delete = False
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        try:
            conn.execute("PRAGMA journal_mode = DELETE")
            switched_to_delete = True
            conn.execute(f"PRAGMA page_size = {PAGE_SIZE}")
        except sqlite3.OperationalError:
            # Another process has the DB open in WAL mode; keep WAL and still VACUUM to reclaim dead pages
            pass
        conn.execute("PRAGMA auto_vacuum = INCREMENTAL")
        conn.execute("VACUUM")
        if switched_to_delete:
            conn.execute("PRAGMA journal_mode = WAL")
        else:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except sqlite3.OperationalError:
        if switched_to_delete:
            try:
                conn.execute("PRAGMA journal_mode = WAL")
            except sqlite3.OperationalError:
                pass
    finally:
        conn.close()


def migrate_database(path: str | Path) -> bool:
    from ..infra.ipc import log as ipc_log

    p = Path(path)
    if not p.is_file():
        return False

    ipc_log(f"scene scout {p.name} | checking database schema...")
    needs_compaction = False
    fallback_created = _file_time_iso(p)

    conn = sqlite3.connect(str(p), timeout=30.0, isolation_level=None)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        current_version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        if current_version > SCHEMA_VERSION:
            meta_now = (
                dict(conn.execute("SELECT key, value FROM db_metadata").fetchall())
                if "db_metadata" in _table_names(conn)
                else {}
            )
            raise UnsupportedSchemaError(
                p,
                current_version,
                updated_by_app=meta_now.get("updated_by_app") or meta_now.get("created_by_app"),
                updated_by_version=meta_now.get("updated_by_version") or meta_now.get("created_by_version"),
            )

        tables = _table_names(conn)
        if not tables.intersection({"processed_videos", "scenes", "scene_embeddings", "image_embeddings"}):
            raise ValueError(f"'{p.name}' is not a valid Scene Scout database (missing core tables).")

        already_v4 = (
            current_version == SCHEMA_VERSION
            and {"processed_videos", "scenes", "video_embeddings", "scene_thumbnails", "db_metadata", "preprocessing_configs", "media_roots"}.issubset(tables)
            and "scene_embeddings" not in tables
            and "image_embeddings" not in tables
        )
        if already_v4:
            conn.execute("BEGIN IMMEDIATE")
            try:
                _rebuild_scene_motion_fk(conn)
                _seed_or_touch_metadata(conn, is_fresh=False, fallback_created_at=fallback_created)
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
            return True

        ipc_log(f"scene scout {p.name} | migrating schema v{current_version} -> v{SCHEMA_VERSION}...")
        needs_compaction = True

        # PRAGMA foreign_keys must be toggled OUTSIDE any active transaction
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("BEGIN IMMEDIATE")
        try:
            # Step 1: If pre-v3 (filepath FK on scene_embeddings, no video_id), normalize to v3 first
            if "scene_embeddings" in tables:
                se_cols = _column_names(conn, "scene_embeddings")
                if "thumbnail" not in se_cols:
                    try:
                        conn.execute("ALTER TABLE scene_embeddings ADD COLUMN thumbnail BLOB")
                    except sqlite3.OperationalError:
                        pass
                    se_cols = _column_names(conn, "scene_embeddings")

                if "video_id" not in se_cols:
                    pv_cols = _column_names(conn, "processed_videos") if "processed_videos" in tables else set()
                    has_mv = "model_version" in pv_cols
                    conn.execute(
                        f"""
                        CREATE TABLE processed_videos_v3 (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            filepath TEXT UNIQUE NOT NULL,
                            modified_at REAL NOT NULL,
                            model_version TEXT DEFAULT '{EMBEDDING_MODEL_VERSION}',
                            status TEXT DEFAULT 'completed'
                        )
                        """
                    )
                    conn.execute(
                        """
                        CREATE TABLE scene_embeddings_v3 (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            video_id INTEGER NOT NULL,
                            scene_index INTEGER NOT NULL,
                            start_time_ms INTEGER NOT NULL,
                            end_time_ms INTEGER NOT NULL,
                            embedding BLOB NOT NULL,
                            thumbnail BLOB
                        )
                        """
                    )
                    if "processed_videos" in tables:
                        if has_mv:
                            conn.execute(
                                """
                                INSERT OR IGNORE INTO processed_videos_v3 (filepath, modified_at, model_version)
                                SELECT filepath, modified_at, model_version FROM processed_videos
                                """
                            )
                        else:
                            conn.execute(
                                """
                                INSERT OR IGNORE INTO processed_videos_v3 (filepath, modified_at)
                                SELECT filepath, modified_at FROM processed_videos
                                """
                            )
                    conn.execute(
                        """
                        INSERT INTO scene_embeddings_v3
                            (video_id, scene_index, start_time_ms, end_time_ms, embedding, thumbnail)
                        SELECT pv.id, se.scene_index, se.start_time_ms, se.end_time_ms, se.embedding, se.thumbnail
                        FROM scene_embeddings se
                        JOIN processed_videos_v3 pv ON se.filepath = pv.filepath
                        """
                    )
                    conn.execute("DROP TABLE scene_embeddings")
                    if "processed_videos" in tables:
                        conn.execute("DROP TABLE processed_videos")
                    conn.execute("ALTER TABLE processed_videos_v3 RENAME TO processed_videos")
                    conn.execute("ALTER TABLE scene_embeddings_v3 RENAME TO scene_embeddings")
                    tables = _table_names(conn)

            # Step 2: Upgrade to v4 structure (no executescript() inside atomic transaction)
            for ddl in (
                """
                CREATE TABLE IF NOT EXISTS db_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS media_roots (
                    root_path TEXT PRIMARY KEY,
                    added_at TEXT NOT NULL
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS preprocessing_configs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_id TEXT NOT NULL,
                    detector TEXT NOT NULL DEFAULT 'unknown',
                    sampling_strategy TEXT NOT NULL DEFAULT 'unknown',
                    sampling_config TEXT NOT NULL DEFAULT '{}',
                    UNIQUE(model_id, detector, sampling_strategy, sampling_config)
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS video_embeddings (
                    video_id INTEGER PRIMARY KEY,
                    scene_count INTEGER NOT NULL,
                    dim INTEGER NOT NULL,
                    dtype TEXT NOT NULL DEFAULT 'float32',
                    embeddings BLOB NOT NULL,
                    FOREIGN KEY (video_id) REFERENCES processed_videos(id) ON DELETE CASCADE
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS scenes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    video_id INTEGER NOT NULL,
                    scene_index INTEGER NOT NULL,
                    start_time_ms INTEGER NOT NULL,
                    end_time_ms INTEGER NOT NULL,
                    FOREIGN KEY (video_id) REFERENCES processed_videos(id) ON DELETE CASCADE
                )
                """,
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_scene_video_scene ON scenes(video_id, scene_index)",
                """
                CREATE TABLE IF NOT EXISTS scene_thumbnails (
                    scene_id INTEGER PRIMARY KEY,
                    thumbnail BLOB NOT NULL,
                    FOREIGN KEY (scene_id) REFERENCES scenes(id) ON DELETE CASCADE
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS index_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    path TEXT UNIQUE NOT NULL,
                    is_directory BOOLEAN NOT NULL DEFAULT 0,
                    recursive BOOLEAN NOT NULL DEFAULT 1,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """,
            ):
                conn.execute(ddl)

            if "processed_videos" not in tables:
                conn.execute(
                    """
                    CREATE TABLE processed_videos (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filepath TEXT UNIQUE NOT NULL,
                        modified_at REAL NOT NULL,
                        status TEXT NOT NULL DEFAULT 'completed',
                        scene_count INTEGER NOT NULL DEFAULT 0,
                        config_id INTEGER REFERENCES preprocessing_configs(id),
                        file_size INTEGER,
                        content_fingerprint TEXT,
                        duration_ms INTEGER,
                        fps REAL,
                        width INTEGER,
                        height INTEGER
                    )
                    """
                )
            else:
                pv_cols = _column_names(conn, "processed_videos")
                additive_cols = [
                    ("status", "TEXT NOT NULL DEFAULT 'completed'"),
                    ("scene_count", "INTEGER NOT NULL DEFAULT 0"),
                    ("config_id", "INTEGER REFERENCES preprocessing_configs(id)"),
                    ("file_size", "INTEGER"),
                    ("content_fingerprint", "TEXT"),
                    ("duration_ms", "INTEGER"),
                    ("fps", "REAL"),
                    ("width", "INTEGER"),
                    ("height", "INTEGER"),
                ]
                for col_name, col_def in additive_cols:
                    if col_name not in pv_cols:
                        conn.execute(f"ALTER TABLE processed_videos ADD COLUMN {col_name} {col_def}")

                pv_cols = _column_names(conn, "processed_videos")
                if "model_version" in pv_cols:
                    distinct_models = conn.execute(
                        "SELECT DISTINCT COALESCE(NULLIF(TRIM(model_version), ''), ?) FROM processed_videos",
                        (EMBEDDING_MODEL_VERSION,),
                    ).fetchall()
                    for (m_id,) in distinct_models:
                        cfg_id = get_or_create_config(conn, m_id, "unknown", "unknown", "{}")
                        conn.execute(
                            """
                            UPDATE processed_videos
                            SET config_id = ?
                            WHERE COALESCE(NULLIF(TRIM(model_version), ''), ?) = ?
                            """,
                            (cfg_id, EMBEDDING_MODEL_VERSION, m_id),
                        )
                    conn.execute("ALTER TABLE processed_videos DROP COLUMN model_version")

            # Migrate scene_embeddings -> scenes + video_embeddings + scene_thumbnails
            if "scene_embeddings" in tables:
                se_cols = _column_names(conn, "scene_embeddings")
                thumb_expr = "thumbnail" if "thumbnail" in se_cols else "NULL"
                video_ids = [r[0] for r in conn.execute("SELECT id FROM processed_videos").fetchall()]
                for vid in video_ids:
                    # Preserve intact v4 scenes/embeddings if an old v3 app created an empty scene_embeddings table
                    if conn.execute("SELECT 1 FROM scenes WHERE video_id = ? LIMIT 1", (vid,)).fetchone() is not None:
                        continue
                    rows = conn.execute(
                        f"""
                        SELECT id, scene_index, start_time_ms, end_time_ms, embedding, {thumb_expr}
                        FROM scene_embeddings
                        WHERE video_id = ?
                        ORDER BY scene_index ASC, id ASC
                        """,
                        (vid,),
                    ).fetchall()
                    seen_indices: set[int] = set()
                    deduped = []
                    for row in rows:
                        if row[1] in seen_indices:
                            continue
                        seen_indices.add(row[1])
                        deduped.append(row)

                    dim = 0
                    for row in deduped:
                        blob = row[4]
                        if blob and len(blob) >= 4 and len(blob) % 4 == 0:
                            dim = len(blob) // 4
                            break

                    scene_inserts: list[tuple[int, int, int, int, int]] = []
                    thumb_inserts: list[tuple[int, bytes]] = []
                    emb_blobs: list[bytes] = []
                    want_len = dim * 4

                    for r_id, _old_idx, start_ms, end_ms, blob, thumb in deduped:
                        if want_len <= 0 or blob is None or len(blob) != want_len:
                            continue
                        new_idx = len(scene_inserts)
                        scene_inserts.append((r_id, vid, new_idx, int(start_ms), int(end_ms)))
                        emb_blobs.append(bytes(blob))
                        if thumb:
                            thumb_inserts.append((r_id, bytes(thumb)))

                    if scene_inserts:
                        conn.executemany(
                            "INSERT OR REPLACE INTO scenes (id, video_id, scene_index, start_time_ms, end_time_ms) VALUES (?, ?, ?, ?, ?)",
                            scene_inserts,
                        )
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO video_embeddings (video_id, scene_count, dim, dtype, embeddings)
                            VALUES (?, ?, ?, 'float32', ?)
                            """,
                            (vid, len(scene_inserts), dim, b"".join(emb_blobs)),
                        )
                    if thumb_inserts:
                        conn.executemany(
                            "INSERT OR REPLACE INTO scene_thumbnails (scene_id, thumbnail) VALUES (?, ?)",
                            thumb_inserts,
                        )
                    conn.execute(
                        "UPDATE processed_videos SET scene_count = ? WHERE id = ?",
                        (len(scene_inserts), vid),
                    )

                conn.execute("DROP TABLE scene_embeddings")

            conn.execute("DROP TABLE IF EXISTS image_embeddings")

            _rebuild_scene_motion_fk(conn)

            # Backfill media_roots from queued directories and indexed video paths
            for (q_path,) in conn.execute("SELECT path FROM index_queue WHERE is_directory = 1").fetchall():
                record_media_root(conn, q_path, explicit_root=q_path)
            for (v_fp,) in conn.execute("SELECT filepath FROM processed_videos").fetchall():
                record_media_root(conn, v_fp)

            _seed_or_touch_metadata(conn, is_fresh=False, fallback_created_at=fallback_created)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.execute("COMMIT")
            ipc_log(f"scene scout {p.name} | v4 schema migration completed successfully")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.OperationalError:
                pass
            raise
        finally:
            try:
                conn.execute("PRAGMA foreign_keys = ON")
            except sqlite3.OperationalError:
                pass
    finally:
        conn.close()

    if needs_compaction:
        _compact_database_v4(p)
    invalidate_embedding_cache(p)
    return True


def run_incremental_vacuum(path: str | Path) -> None:
    """Reclaim freed pages under ``auto_vacuum = INCREMENTAL`` if enabled."""
    raw_conn = sqlite3.connect(str(path), timeout=10.0, isolation_level=None)
    try:
        if raw_conn.execute("PRAGMA auto_vacuum").fetchone()[0] == 2:
            raw_conn.execute("PRAGMA incremental_vacuum")
    except sqlite3.OperationalError:
        pass
    finally:
        raw_conn.close()


def open_database(path: str | Path) -> DatabaseInfo:
    from ..infra.ipc import log as ipc_log

    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Database file not found: {p}")
    ipc_log(f"scene scout {p.name} | opening database...")
    ensure_v4_ready(p)
    info = database_info(p)
    ipc_log(f"scene scout {p.name} | database ready ({info.video_count} videos, {info.scene_count} scenes)")
    return info


# ---------------------------------------------------------------------------
# Queries & CRUD
# ---------------------------------------------------------------------------


def database_info(path: str | Path) -> DatabaseInfo:
    p = Path(path)
    if not p.is_file():
        return DatabaseInfo(
            name=p.stem,
            path=str(p),
            video_count=0,
            scene_count=0,
            size_bytes=0,
            model_version=EMBEDDING_MODEL_VERSION,
            schema_version=SCHEMA_VERSION,
            schema_status="ready",
        )

    inspected = inspect_database_schema(p)
    meta = inspected["meta"]
    sv = int(inspected["schema_version"]) if inspected["schema_version"] else SCHEMA_VERSION
    if inspected["status"] == "ready" and meta.get("schema_version", "").isdigit():
        sv = int(meta["schema_version"])

    schema_status = inspected["status"] if inspected["status"] in ("ready", "needs_upgrade", "unsupported_newer") else "ready"

    return DatabaseInfo(
        name=p.stem,
        path=str(p),
        video_count=int(inspected["video_count"]),
        scene_count=int(inspected["scene_count"]),
        size_bytes=p.stat().st_size if p.is_file() else 0,
        model_version=str(inspected["model_version"]),
        created_at=meta.get("created_at"),
        database_id=meta.get("database_id"),
        schema_version=sv,
        schema_status=schema_status,
        created_by_app=meta.get("created_by_app"),
        created_by_version=meta.get("created_by_version"),
        updated_by_app=meta.get("updated_by_app"),
        updated_by_version=meta.get("updated_by_version"),
        updated_at=meta.get("updated_at"),
        media_roots=list(inspected["roots"]),
    )


def list_databases(root: str | Path | None = None) -> list[DatabaseInfo]:
    return [database_info(p) for p in list_db_paths(root)]


def delete_database(name: str, root: str | Path | None = None) -> bool:
    """Remove a database and the WAL sidecars SQLite leaves beside it."""
    path = db_path(name, root)
    invalidate_embedding_cache(path)
    if not path.is_file():
        return False
    path.unlink()
    for sidecar in (path.with_suffix(path.suffix + "-wal"), path.with_suffix(path.suffix + "-shm")):
        if sidecar.exists():
            sidecar.unlink()
    return True


def _parse_sampling_config(raw: Optional[str]) -> Optional[dict[str, Any]]:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None


def list_videos(path: str | Path) -> list[IndexedVideo]:
    with connect_ro(path) as conn:
        uv = int(conn.execute("PRAGMA user_version").fetchone()[0])
        tables = _table_names(conn)
        if uv > SCHEMA_VERSION:
            meta = dict(conn.execute("SELECT key, value FROM db_metadata").fetchall()) if "db_metadata" in tables else {}
            raise UnsupportedSchemaError(
                path,
                uv,
                updated_by_app=meta.get("updated_by_app") or meta.get("created_by_app"),
                updated_by_version=meta.get("updated_by_version") or meta.get("created_by_version"),
            )
        if "processed_videos" not in tables:
            return []
        pv_cols = _column_names(conn, "processed_videos")
        if "scene_count" in pv_cols and "preprocessing_configs" in tables:
            rows = conn.execute(
                """
                SELECT v.id, v.filepath, v.status, v.modified_at, v.scene_count,
                       pc.model_id, pc.detector, pc.sampling_strategy, pc.sampling_config,
                       v.file_size, v.content_fingerprint, v.duration_ms, v.fps, v.width, v.height
                FROM processed_videos v
                LEFT JOIN preprocessing_configs pc ON pc.id = v.config_id
                ORDER BY v.id DESC
                """
            ).fetchall()
            return [
                IndexedVideo(
                    id=r[0],
                    filepath=r[1],
                    status=r[2],
                    modified_at=r[3],
                    scene_count=r[4],
                    model_id=r[5],
                    detector=r[6],
                    sampling_strategy=r[7],
                    sampling_config=_parse_sampling_config(r[8]),
                    file_size=r[9],
                    content_fingerprint=r[10],
                    duration_ms=r[11],
                    fps=r[12],
                    width=r[13],
                    height=r[14],
                )
                for r in rows
            ]

        # Legacy un-migrated v3 fallback
        scene_tbl = "scenes" if "scenes" in tables else "scene_embeddings"
        rows = conn.execute(
            f"""
            SELECT v.id, v.filepath, v.status, v.modified_at, COUNT(s.id)
            FROM processed_videos v
            LEFT JOIN {scene_tbl} s ON s.video_id = v.id
            GROUP BY v.id
            ORDER BY v.id DESC
            """
        ).fetchall()
        return [
            IndexedVideo(id=r[0], filepath=r[1], status=r[2], modified_at=r[3], scene_count=r[4])
            for r in rows
        ]


def find_video(path: str | Path, filepath: str) -> Optional[IndexedVideo]:
    for video in list_videos(path):
        if video.filepath == str(filepath):
            return video
    return None


def remove_video(
    path: str | Path,
    video_id: int | None = None,
    filepath: str | None = None,
) -> bool:
    p = Path(path)
    ensure_v4_ready(p)
    removed = False
    with connect(p) as conn:
        resolved_id = video_id
        resolved_path = filepath
        if resolved_id is None and resolved_path is not None:
            row = conn.execute(
                "SELECT id FROM processed_videos WHERE filepath = ?", (str(resolved_path),)
            ).fetchone()
            if not row:
                return False
            resolved_id = int(row[0])
        elif resolved_id is not None and resolved_path is None:
            row = conn.execute(
                "SELECT filepath FROM processed_videos WHERE id = ?", (resolved_id,)
            ).fetchone()
            if row:
                resolved_path = str(row[0])

        if resolved_id is None:
            return False

        # Explicit deletes alongside ON DELETE CASCADE keep behavior identical even if FK was toggled
        conn.execute(
            "DELETE FROM scene_thumbnails WHERE scene_id IN (SELECT id FROM scenes WHERE video_id = ?)",
            (resolved_id,),
        )
        if has_motion_table(conn):
            conn.execute(
                "DELETE FROM scene_motion WHERE scene_id IN (SELECT id FROM scenes WHERE video_id = ?)",
                (resolved_id,),
            )
        conn.execute("DELETE FROM video_embeddings WHERE video_id = ?", (resolved_id,))
        conn.execute("DELETE FROM scenes WHERE video_id = ?", (resolved_id,))
        if resolved_path is not None:
            conn.execute("DELETE FROM index_queue WHERE path = ?", (str(resolved_path),))
        cur = conn.execute("DELETE FROM processed_videos WHERE id = ?", (resolved_id,))
        removed = cur.rowcount > 0
        if removed:
            _touch_updated_metadata(conn)

    if removed:
        run_incremental_vacuum(p)
        invalidate_embedding_cache(p)
    return removed


def upsert_video(
    path: str | Path,
    filepath: str,
    modified_at: float,
    status: str = "indexing",
    *,
    model_id: str = EMBEDDING_MODEL_VERSION,
    detector: str = "unknown",
    sampling_strategy: str = "unknown",
    sampling_config: Optional[dict[str, Any]] = None,
    file_size: Optional[int] = None,
    content_fingerprint: Optional[str] = None,
    duration_ms: Optional[int] = None,
    fps: Optional[float] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> int:
    """Insert or reset a video row and return its id.

    Re-indexing clears old scenes, embeddings, thumbnails, and motion first so
    data from two runs never mixes.
    """
    p = Path(path)
    ensure_v4_ready(p)
    with connect(p) as conn:
        config_id = get_or_create_config(
            conn,
            model_id=model_id,
            detector=detector,
            sampling_strategy=sampling_strategy,
            sampling_config=sampling_config,
        )
        conn.execute(
            """
            INSERT INTO processed_videos (
                filepath, modified_at, status, scene_count, config_id,
                file_size, content_fingerprint, duration_ms, fps, width, height
            )
            VALUES (?, ?, ?, 0, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(filepath) DO UPDATE SET
                modified_at = excluded.modified_at,
                status = excluded.status,
                scene_count = 0,
                config_id = excluded.config_id,
                file_size = COALESCE(excluded.file_size, processed_videos.file_size),
                content_fingerprint = COALESCE(excluded.content_fingerprint, processed_videos.content_fingerprint),
                duration_ms = COALESCE(excluded.duration_ms, processed_videos.duration_ms),
                fps = COALESCE(excluded.fps, processed_videos.fps),
                width = COALESCE(excluded.width, processed_videos.width),
                height = COALESCE(excluded.height, processed_videos.height)
            """,
            (
                str(filepath),
                modified_at,
                status,
                config_id,
                file_size,
                content_fingerprint,
                duration_ms,
                fps,
                width,
                height,
            ),
        )
        video_id = int(
            conn.execute("SELECT id FROM processed_videos WHERE filepath = ?", (str(filepath),)).fetchone()[0]
        )
        conn.execute(
            "DELETE FROM scene_thumbnails WHERE scene_id IN (SELECT id FROM scenes WHERE video_id = ?)",
            (video_id,),
        )
        if has_motion_table(conn):
            conn.execute(
                "DELETE FROM scene_motion WHERE scene_id IN (SELECT id FROM scenes WHERE video_id = ?)",
                (video_id,),
            )
        conn.execute("DELETE FROM video_embeddings WHERE video_id = ?", (video_id,))
        conn.execute("DELETE FROM scenes WHERE video_id = ?", (video_id,))
        record_media_root(conn, str(filepath))
        _touch_updated_metadata(conn)

    invalidate_embedding_cache(p)
    return video_id


def insert_video_scenes(
    path: str | Path,
    video_id: int,
    scene_ranges: list[tuple[int, int]],
    vectors,
    thumbnails: Optional[list[Optional[bytes]]] = None,
) -> None:
    """Write ``scenes``, contiguous ``video_embeddings``, and ``scene_thumbnails`` in one transaction."""
    import numpy as np

    mat = np.ascontiguousarray(vectors, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat.reshape(len(scene_ranges), -1) if scene_ranges else mat.reshape(0, 0)

    count = len(scene_ranges)
    if mat.shape[0] != count:
        raise ValueError(f"Scene count ({count}) does not match embedding matrix rows ({mat.shape[0]})")
    dim = int(mat.shape[1]) if count > 0 else 0

    p = Path(path)
    with connect(p) as conn:
        if count > 0:
            conn.executemany(
                "INSERT INTO scenes (video_id, scene_index, start_time_ms, end_time_ms) VALUES (?, ?, ?, ?)",
                [(video_id, idx, int(s_ms), int(e_ms)) for idx, (s_ms, e_ms) in enumerate(scene_ranges)],
            )
            if dim > 0:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO video_embeddings (video_id, scene_count, dim, dtype, embeddings)
                    VALUES (?, ?, ?, 'float32', ?)
                    """,
                    (video_id, count, dim, mat.tobytes()),
                )
            if thumbnails:
                id_by_idx = dict(
                    conn.execute(
                        "SELECT scene_index, id FROM scenes WHERE video_id = ?", (video_id,)
                    ).fetchall()
                )
                thumb_rows = [
                    (id_by_idx[idx], thumb)
                    for idx, thumb in enumerate(thumbnails)
                    if thumb is not None and idx in id_by_idx
                ]
                if thumb_rows:
                    conn.executemany(
                        "INSERT OR REPLACE INTO scene_thumbnails (scene_id, thumbnail) VALUES (?, ?)",
                        thumb_rows,
                    )
        conn.execute("UPDATE processed_videos SET scene_count = ? WHERE id = ?", (count, video_id))
        _touch_updated_metadata(conn)

    invalidate_embedding_cache(p)


def insert_scenes(
    path: str | Path,
    video_id: int,
    scenes: list[tuple[int, int, int, bytes, bytes | None]],
) -> None:
    """Backwards-compatible bulk scene writer taking ``(scene_index, start_ms, end_ms, embedding, thumbnail)``."""
    if not scenes:
        with connect(path) as conn:
            conn.execute("UPDATE processed_videos SET scene_count = 0 WHERE id = ?", (video_id,))
        invalidate_embedding_cache(path)
        return

    ordered = sorted(scenes, key=lambda r: r[0])
    first_blob = ordered[0][3]
    if len(first_blob) % 4 != 0:
        raise ValueError(f"Embedding byte length ({len(first_blob)}) must be a multiple of 4 (float32).")
    dim = len(first_blob) // 4
    count = len(ordered)

    p = Path(path)
    with connect(p) as conn:
        conn.executemany(
            "INSERT INTO scenes (video_id, scene_index, start_time_ms, end_time_ms) VALUES (?, ?, ?, ?)",
            [(video_id, idx, int(r[1]), int(r[2])) for idx, r in enumerate(ordered)],
        )
        if dim > 0:
            conn.execute(
                """
                INSERT OR REPLACE INTO video_embeddings (video_id, scene_count, dim, dtype, embeddings)
                VALUES (?, ?, ?, 'float32', ?)
                """,
                (video_id, count, dim, b"".join(bytes(r[3]) for r in ordered)),
            )
        id_by_idx = dict(
            conn.execute("SELECT scene_index, id FROM scenes WHERE video_id = ?", (video_id,)).fetchall()
        )
        thumb_rows = [
            (id_by_idx[idx], bytes(r[4]))
            for idx, r in enumerate(ordered)
            if r[4] is not None and idx in id_by_idx
        ]
        if thumb_rows:
            conn.executemany(
                "INSERT OR REPLACE INTO scene_thumbnails (scene_id, thumbnail) VALUES (?, ?)",
                thumb_rows,
            )
        conn.execute("UPDATE processed_videos SET scene_count = ? WHERE id = ?", (count, video_id))
        _touch_updated_metadata(conn)

    invalidate_embedding_cache(p)


def mark_video_complete(path: str | Path, video_id: int) -> None:
    with connect(path) as conn:
        conn.execute("UPDATE processed_videos SET status = 'completed' WHERE id = ?", (video_id,))
        _touch_updated_metadata(conn)
    invalidate_embedding_cache(path)


def insert_scene_motion(
    path: str | Path,
    video_id: int,
    motion: dict[int, tuple[Optional[dict], Optional[dict]]],
    version: int,
) -> int:
    """Store ``{scene_index: (start_raw, end_raw)}`` for one video. Returns rows written."""
    p = Path(path)
    ensure_v4_ready(p)
    with connect(p) as conn:
        conn.executescript(MOTION_SCHEMA)
        ids = dict(
            conn.execute("SELECT scene_index, id FROM scenes WHERE video_id = ?", (video_id,)).fetchall()
        )
        rows = [
            (ids[index], version, json.dumps(start) if start else None, json.dumps(end) if end else None)
            for index, (start, end) in motion.items()
            if index in ids
        ]
        conn.executemany(
            "INSERT OR REPLACE INTO scene_motion (scene_id, motion_version, start_raw, end_raw) VALUES (?, ?, ?, ?)",
            rows,
        )
        _touch_updated_metadata(conn)
        return len(rows)


def videos_missing_motion(path: str | Path, version: int) -> list[tuple[int, str, list[tuple[int, int, int]]]]:
    """Completed videos with scenes lacking current motion: ``(video_id, filepath, [(scene_index, start_ms, end_ms)])``."""
    p = Path(path)
    ensure_v4_ready(p)
    with connect(p) as conn:
        conn.executescript(MOTION_SCHEMA)
        rows = conn.execute(
            """
            SELECT v.id, v.filepath, s.scene_index, s.start_time_ms, s.end_time_ms,
                   (m.scene_id IS NOT NULL AND m.motion_version = ?) AS current
            FROM processed_videos v
            JOIN scenes s ON s.video_id = v.id
            LEFT JOIN scene_motion m ON m.scene_id = s.id
            WHERE v.status = 'completed'
            ORDER BY v.id, s.start_time_ms
            """,
            (version,),
        ).fetchall()

    by_video: dict[int, tuple[str, list[tuple[int, int, int]], bool]] = {}
    for video_id, filepath, index, start, end, current in rows:
        entry = by_video.setdefault(video_id, (filepath, [], False))
        entry[1].append((index, start, end))
        if not current:
            by_video[video_id] = (entry[0], entry[1], True)
    return [(vid, fp, scs) for vid, (fp, scs, stale) in by_video.items() if stale]


# ---------------------------------------------------------------------------
# Smart Media Relinking (media_roots relative path + basename + fingerprint)
# ---------------------------------------------------------------------------


def _derive_relative_candidates(old_filepath: str, roots: list[str]) -> list[str]:
    """Derive candidate relative paths for ``old_filepath`` against stored ``media_roots``."""
    norm_fp = _norm_slash_path(old_filepath)
    candidates: list[str] = []
    # Longest matching root first
    matching = sorted(
        (r for r in roots if _path_is_under_root(norm_fp, r)),
        key=lambda r: len(r),
        reverse=True,
    )
    for root in matching:
        rel = norm_fp[len(root.rstrip("/")) + 1 :]
        if rel and rel not in candidates:
            candidates.append(rel)
        # Also include <root_tail>/<rel> in case user pointed at the parent of root
        root_tail = PurePosixPath(root.rstrip("/")).name
        if root_tail:
            prefixed = f"{root_tail}/{rel}"
            if prefixed not in candidates:
                candidates.append(prefixed)
    return candidates


def relink_database_paths(
    path: str | Path,
    search_roots: list[str | Path],
    *,
    verify_hash: bool = True,
) -> dict[str, Any]:
    """Relink missing ``processed_videos`` rows under ``search_roots``."""
    p = Path(path).resolve()
    ensure_v4_ready(p)

    dirs = [Path(sr).expanduser().resolve() for sr in search_roots if Path(sr).expanduser().is_dir()]
    with connect(p) as conn:
        roots = [r[0] for r in conn.execute("SELECT root_path FROM media_roots").fetchall()]
        videos = conn.execute(
            "SELECT id, filepath, file_size, content_fingerprint FROM processed_videos ORDER BY id"
        ).fetchall()
        existing_paths = {str(Path(r[1])) for r in videos}

        missing = [r for r in videos if not Path(r[1]).is_file()]
        if not missing or not dirs:
            return {
                "database": str(p),
                "checked": len(videos),
                "missing": len(missing),
                "relinked": 0,
                "unresolved": len(missing),
                "updated": [],
            }

        # Lazy index of candidate files by size and lowercase basename across search_roots
        walked = False
        by_size: dict[int, list[Path]] = {}
        by_name: dict[str, list[Path]] = {}

        def ensure_walked() -> None:
            nonlocal walked
            if walked:
                return
            walked = True
            for base in dirs:
                for root_dir, _subdirs, files in os.walk(base):
                    for fn in files:
                        cand = Path(root_dir) / fn
                        try:
                            sz = cand.stat().st_size
                        except OSError:
                            continue
                        by_size.setdefault(sz, []).append(cand)
                        by_name.setdefault(fn.lower(), []).append(cand)

        def matches_video(cand: Path, exp_size: Optional[int], exp_fp: Optional[str]) -> bool:
            if not cand.is_file():
                return False
            if verify_hash and (exp_size is not None or exp_fp):
                return verify_content_fingerprint(cand, exp_size, exp_fp)
            return True

        updated: list[dict[str, Any]] = []
        for vid_id, old_fp, f_size, c_fp in missing:
            chosen: Optional[Path] = None
            rel_candidates = _derive_relative_candidates(old_fp, roots)
            basename = Path(_norm_slash_path(old_fp)).name

            # 1. Try media_roots relative paths directly under each search_root
            for base in dirs:
                for rel in rel_candidates:
                    cand = base.joinpath(*rel.split("/"))
                    if matches_video(cand, f_size, c_fp):
                        chosen = cand.resolve()
                        break
                if chosen:
                    break

            # 2. Try direct basename in search_root
            if chosen is None:
                for base in dirs:
                    cand = base / basename
                    if matches_video(cand, f_size, c_fp):
                        chosen = cand.resolve()
                        break

            # 3. Walk search_roots: match exact basename first, then content fingerprint (handles renames!)
            if chosen is None:
                ensure_walked()
                for cand in by_name.get(basename.lower(), []):
                    if matches_video(cand, f_size, c_fp):
                        chosen = cand.resolve()
                        break

            if chosen is None and f_size is not None and c_fp:
                ensure_walked()
                for cand in by_size.get(int(f_size), []):
                    if verify_content_fingerprint(cand, f_size, c_fp):
                        chosen = cand.resolve()
                        break

            if chosen is not None:
                new_str = str(chosen)
                if new_str in existing_paths and new_str != old_fp:
                    continue
                try:
                    conn.execute(
                        "UPDATE processed_videos SET filepath = ? WHERE id = ?",
                        (new_str, vid_id),
                    )
                    existing_paths.discard(old_fp)
                    existing_paths.add(new_str)
                    record_media_root(conn, new_str)
                    updated.append({"id": vid_id, "oldPath": old_fp, "newPath": new_str})
                except sqlite3.IntegrityError:
                    continue

        if updated:
            _touch_updated_metadata(conn)

    if updated:
        invalidate_embedding_cache(p)

    return {
        "database": str(p),
        "checked": len(videos),
        "missing": len(missing),
        "relinked": len(updated),
        "unresolved": len(missing) - len(updated),
        "updated": updated,
    }
