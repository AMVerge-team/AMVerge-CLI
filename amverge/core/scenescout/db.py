"""SQLite storage for Scene Scout.

The schema is deliberately compatible with the upstream scene-scout project
(https://github.com/Mark-Shun/scene-scout) so a database produced by either tool
opens in the other. Keep it that way: diverging here means a user's existing
databases stop working when they move between the standalone tool and the app.

Only stdlib imports at module scope. Listing databases and reporting status must
work on a machine that has never installed torch.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from .paths import db_path, list_db_paths, ensure_root
from .types import DatabaseInfo, IndexedVideo, EMBEDDING_MODEL_VERSION

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS processed_videos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filepath TEXT UNIQUE NOT NULL,
    modified_at REAL NOT NULL,
    model_version TEXT DEFAULT '{EMBEDDING_MODEL_VERSION}',
    status TEXT DEFAULT 'completed'
);

CREATE TABLE IF NOT EXISTS scene_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id INTEGER NOT NULL,
    scene_index INTEGER NOT NULL,
    start_time_ms INTEGER NOT NULL,
    end_time_ms INTEGER NOT NULL,
    embedding BLOB NOT NULL,
    thumbnail BLOB,
    FOREIGN KEY (video_id) REFERENCES processed_videos(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_scene_video_id ON scene_embeddings(video_id);

CREATE TABLE IF NOT EXISTS image_embeddings (
    filepath TEXT PRIMARY KEY,
    modified_at REAL NOT NULL,
    embedding BLOB NOT NULL,
    model_version TEXT DEFAULT '{EMBEDDING_MODEL_VERSION}',
    file_type TEXT DEFAULT 'image'
);

CREATE TABLE IF NOT EXISTS index_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE NOT NULL,
    is_directory BOOLEAN NOT NULL DEFAULT 0,
    recursive BOOLEAN NOT NULL DEFAULT 1,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


@contextmanager
def connect(path: str | Path, timeout: float = 10.0) -> Iterator[sqlite3.Connection]:
    """Open a database with the pragmas indexing throughput depends on.

    WAL plus ``synchronous=NORMAL`` is the difference between a usable index run
    and one that fsyncs per scene. The tradeoff is losing the last transactions
    on a hard power cut, which for a rebuildable cache is the right trade.
    """
    conn = sqlite3.connect(str(path), timeout=timeout)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        yield conn
        conn.commit()
    finally:
        conn.close()


def create_database(name: str, root: str | Path | None = None) -> Path:
    path = db_path(name, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect(path) as conn:
        conn.executescript(SCHEMA)
        conn.execute("PRAGMA user_version = 3")
    return path


def database_info(path: str | Path) -> DatabaseInfo:
    p = Path(path)
    videos = scenes = 0
    model_version = EMBEDDING_MODEL_VERSION
    if p.is_file():
        try:
            with connect(p) as conn:
                videos = conn.execute("SELECT COUNT(*) FROM processed_videos").fetchone()[0]
                scenes = conn.execute("SELECT COUNT(*) FROM scene_embeddings").fetchone()[0]
                row = conn.execute("SELECT model_version FROM processed_videos LIMIT 1").fetchone()
                if row and row[0]:
                    model_version = str(row[0])
        except sqlite3.DatabaseError:
            pass

    return DatabaseInfo(
        name=p.stem,
        path=str(p),
        video_count=videos,
        scene_count=scenes,
        size_bytes=p.stat().st_size if p.is_file() else 0,
        model_version=model_version,
    )


def is_valid_database(path: str | Path) -> bool:
    p = Path(path)
    if not p.is_file():
        return False
    try:
        with connect(p) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}
            return bool(tables.intersection({"processed_videos", "scene_embeddings", "image_embeddings"}))
    except Exception:
        return False


def migrate_database(path: str | Path) -> bool:
    from ..infra.ipc import log as ipc_log

    p = Path(path)
    if not p.is_file():
        return False

    ipc_log(f"scene scout {p.name} | checking database schema...")
    with connect(p) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        if not tables.intersection({"processed_videos", "scene_embeddings", "image_embeddings"}):
            raise ValueError(f"'{p.name}' is not a valid Scene Scout database (missing core tables).")

        has_scene_embeddings = "scene_embeddings" in tables

        if has_scene_embeddings:
            cursor.execute("PRAGMA table_info(scene_embeddings)")
            cols = {row[1] for row in cursor.fetchall()}
            if "thumbnail" not in cols:
                ipc_log(f"scene scout {p.name} | upgrading schema: adding thumbnail column...")
                try:
                    conn.execute("ALTER TABLE scene_embeddings ADD COLUMN thumbnail BLOB")
                except sqlite3.OperationalError:
                    pass

        conn.execute("""
            CREATE TABLE IF NOT EXISTS index_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE NOT NULL,
                is_directory BOOLEAN NOT NULL DEFAULT 0,
                recursive BOOLEAN NOT NULL DEFAULT 1,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.execute(f"""
            CREATE TABLE IF NOT EXISTS image_embeddings (
                filepath TEXT PRIMARY KEY,
                modified_at REAL NOT NULL,
                embedding BLOB NOT NULL,
                model_version TEXT DEFAULT '{EMBEDDING_MODEL_VERSION}',
                file_type TEXT DEFAULT 'image'
            )
        """)

        needs_v3 = False
        if has_scene_embeddings:
            cursor.execute("PRAGMA table_info(scene_embeddings)")
            cols = {row[1] for row in cursor.fetchall()}
            if "video_id" not in cols:
                needs_v3 = True

        if needs_v3:
            ipc_log(f"scene scout {p.name} | migrating schema to v3: converting to relational video/scene tables...")
            conn.execute("PRAGMA foreign_keys = OFF")
            try:
                conn.execute(f"""
                    CREATE TABLE processed_videos_new (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        filepath TEXT UNIQUE NOT NULL,
                        modified_at REAL NOT NULL,
                        model_version TEXT DEFAULT '{EMBEDDING_MODEL_VERSION}',
                        status TEXT DEFAULT 'completed'
                    )
                """)
                conn.execute("""
                    CREATE TABLE scene_embeddings_new (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        video_id INTEGER NOT NULL,
                        scene_index INTEGER NOT NULL,
                        start_time_ms INTEGER NOT NULL,
                        end_time_ms INTEGER NOT NULL,
                        embedding BLOB NOT NULL,
                        thumbnail BLOB,
                        FOREIGN KEY (video_id) REFERENCES processed_videos_new(id) ON DELETE CASCADE
                    )
                """)
                conn.execute("""
                    INSERT INTO processed_videos_new (filepath, modified_at, model_version)
                    SELECT filepath, modified_at, model_version FROM processed_videos
                """)
                conn.execute("""
                    INSERT INTO scene_embeddings_new (video_id, scene_index, start_time_ms, end_time_ms, embedding, thumbnail)
                    SELECT pv_new.id, se.scene_index, se.start_time_ms, se.end_time_ms, se.embedding, se.thumbnail
                    FROM scene_embeddings se
                    JOIN processed_videos_new pv_new ON se.filepath = pv_new.filepath
                """)
                conn.execute("DROP TABLE scene_embeddings")
                conn.execute("DROP TABLE processed_videos")
                conn.execute("ALTER TABLE processed_videos_new RENAME TO processed_videos")
                conn.execute("ALTER TABLE scene_embeddings_new RENAME TO scene_embeddings")
                conn.execute("CREATE INDEX idx_scene_video_id ON scene_embeddings(video_id)")
                ipc_log(f"scene scout {p.name} | v3 schema migration completed successfully")
            finally:
                conn.execute("PRAGMA foreign_keys = ON")

        conn.execute("PRAGMA user_version = 3")
        return True


def open_database(path: str | Path) -> DatabaseInfo:
    from ..infra.ipc import log as ipc_log

    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Database file not found: {p}")
    ipc_log(f"scene scout {p.name} | opening database...")
    if not is_valid_database(p):
        raise ValueError(f"'{p.name}' is not a valid Scene Scout database.")
    migrate_database(p)
    info = database_info(p)
    ipc_log(f"scene scout {p.name} | database ready ({info.video_count} videos, {info.scene_count} scenes)")
    return info


def list_databases(root: str | Path | None = None) -> list[DatabaseInfo]:
    return [database_info(p) for p in list_db_paths(root)]


def delete_database(name: str, root: str | Path | None = None) -> bool:
    """Remove a database and the WAL sidecars SQLite leaves beside it."""
    path = db_path(name, root)
    if not path.is_file():
        return False
    path.unlink()
    for sidecar in (path.with_suffix(path.suffix + "-wal"), path.with_suffix(path.suffix + "-shm")):
        if sidecar.exists():
            sidecar.unlink()
    return True


def list_videos(path: str | Path) -> list[IndexedVideo]:
    with connect(path) as conn:
        rows = conn.execute(
            """
            SELECT v.id, v.filepath, v.status, v.modified_at, COUNT(s.id)
            FROM processed_videos v
            LEFT JOIN scene_embeddings s ON s.video_id = v.id
            GROUP BY v.id
            ORDER BY v.id DESC
            """
        ).fetchall()

    return [
        IndexedVideo(id=r[0], filepath=r[1], status=r[2], modified_at=r[3], scene_count=r[4])
        for r in rows
    ]


def remove_video(path: str | Path, video_id: int) -> bool:
    """Drop a video and its scenes. The FK cascade handles the embeddings."""
    with connect(path) as conn:
        cur = conn.execute("DELETE FROM processed_videos WHERE id = ?", (video_id,))
        return cur.rowcount > 0


def find_video(path: str | Path, filepath: str) -> Optional[IndexedVideo]:
    with connect(path) as conn:
        row = conn.execute(
            """
            SELECT v.id, v.filepath, v.status, v.modified_at, COUNT(s.id)
            FROM processed_videos v
            LEFT JOIN scene_embeddings s ON s.video_id = v.id
            WHERE v.filepath = ?
            GROUP BY v.id
            """,
            (str(filepath),),
        ).fetchone()

    if not row:
        return None
    return IndexedVideo(id=row[0], filepath=row[1], status=row[2], modified_at=row[3], scene_count=row[4])


# --------------------------------------------------------------------------
# Write path, used by indexing.py
# --------------------------------------------------------------------------

def upsert_video(path: str | Path, filepath: str, modified_at: float, status: str = "indexing") -> int:
    """Insert or reset a video row and return its id.

    Re-indexing clears the old scenes first: leaving them would mix embeddings
    from two runs, and with a changed model version those are not comparable.
    """
    with connect(path) as conn:
        conn.execute(
            """
            INSERT INTO processed_videos (filepath, modified_at, model_version, status)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(filepath) DO UPDATE SET modified_at = excluded.modified_at,
                                                model_version = excluded.model_version,
                                                status = excluded.status
            """,
            (str(filepath), modified_at, EMBEDDING_MODEL_VERSION, status),
        )
        video_id = conn.execute(
            "SELECT id FROM processed_videos WHERE filepath = ?", (str(filepath),)
        ).fetchone()[0]
        conn.execute("DELETE FROM scene_embeddings WHERE video_id = ?", (video_id,))
        return int(video_id)


def insert_scenes(path: str | Path, video_id: int, scenes: list[tuple[int, int, int, bytes, bytes | None]]) -> None:
    """Bulk-insert scenes as ``(scene_index, start_ms, end_ms, embedding, thumbnail)``."""
    with connect(path) as conn:
        conn.executemany(
            """
            INSERT INTO scene_embeddings
                (video_id, scene_index, start_time_ms, end_time_ms, embedding, thumbnail)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [(video_id, *s) for s in scenes],
        )


def mark_video_complete(path: str | Path, video_id: int) -> None:
    with connect(path) as conn:
        conn.execute("UPDATE processed_videos SET status = 'completed' WHERE id = ?", (video_id,))
