"""Comprehensive tests for Scene Scout database schema v4, migrations, cache,
WebP thumbnails, content fingerprints, media roots, and smart relinking."""

from __future__ import annotations

import io
import os
import sqlite3
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from amverge.core.scenescout import db as scoutdb
from amverge.core.scenescout import embedding as scout_embedding
from amverge.core.scenescout import indexing
from amverge.core.scenescout import motion
from amverge.core.scenescout.cache import EMBEDDING_CACHE, EmbeddingCache
from amverge.core.scenescout.fingerprint import (
    FINGERPRINT_ALGO,
    compute_content_fingerprint,
    verify_content_fingerprint,
)
from amverge.core.scenescout.search import search
from amverge.core.scenescout.types import (
    SCHEMA_VERSION,
    SchemaUpgradeRequiredError,
    SearchOptions,
    UnsupportedSchemaError,
)


def _unit_vec(seed: int, dim: int = 8) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_fresh_v4_database_pragmas_and_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AMVERGE_CALLER_APP", "amverge")
    monkeypatch.setenv("AMVERGE_CALLER_VERSION", "2.0.5")

    db_file = scoutdb.create_database("Anime_Season", root=tmp_path)
    info = scoutdb.database_info(db_file)

    assert info.schema_version == SCHEMA_VERSION
    assert info.created_by_app == "amverge"
    assert info.created_by_version == "2.0.5"
    assert info.updated_by_app == "amverge"
    assert info.updated_by_version == "2.0.5"
    assert info.created_at is not None
    assert info.database_id is not None and len(info.database_id) == 36

    with sqlite3.connect(str(db_file)) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 4
        assert conn.execute("PRAGMA page_size").fetchone()[0] == 16384
        assert conn.execute("PRAGMA auto_vacuum").fetchone()[0] == 2  # INCREMENTAL
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert "image_embeddings" not in tables
        assert {"db_metadata", "media_roots", "preprocessing_configs", "processed_videos", "video_embeddings", "scenes", "scene_thumbnails", "index_queue", "scene_motion"}.issubset(tables)
        pv_cols = {r[1] for r in conn.execute("PRAGMA table_info(processed_videos)").fetchall()}
        assert "model_version" not in pv_cols
        assert {"scene_count", "config_id", "file_size", "content_fingerprint", "duration_ms", "fps", "width", "height"}.issubset(pv_cols)


def test_default_cli_caller_identity(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("AMVERGE_CALLER_APP", raising=False)
    monkeypatch.delenv("AMVERGE_CALLER_VERSION", raising=False)

    db_file = scoutdb.create_database("cli_made", root=tmp_path)
    info = scoutdb.database_info(db_file)
    assert info.created_by_app == "amverge-cli"
    assert info.created_by_version is not None


def test_forward_compat_guard_rejects_newer_schema(tmp_path: Path):
    db_file = tmp_path / "future_v9.scoutdb"
    with sqlite3.connect(str(db_file)) as conn:
        conn.execute("CREATE TABLE processed_videos (id INTEGER PRIMARY KEY, filepath TEXT UNIQUE, modified_at REAL)")
        conn.execute("PRAGMA user_version = 9")
        conn.commit()

    with pytest.raises(ValueError, match="v9"):
        scoutdb.migrate_database(db_file)

    # Must NEVER downgrade PRAGMA user_version
    with sqlite3.connect(str(db_file)) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 9


def test_v3_to_v4_migration_preserves_scenes_thumbnails_and_scene_motion_fk(tmp_path: Path):
    db_file = tmp_path / "legacy_v3.scoutdb"
    q = _unit_vec(10)
    v2 = _unit_vec(11)
    jpg_buf = io.BytesIO()
    Image.new("RGB", (320, 180), (120, 80, 200)).save(jpg_buf, format="JPEG", quality=80)
    jpg_bytes = jpg_buf.getvalue()

    with sqlite3.connect(str(db_file)) as conn:
        conn.executescript(
            """
            CREATE TABLE image_embeddings (filepath TEXT PRIMARY KEY, modified_at REAL, embedding BLOB, model_version TEXT, file_type TEXT);
            CREATE TABLE processed_videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filepath TEXT UNIQUE NOT NULL,
                modified_at REAL NOT NULL,
                model_version TEXT DEFAULT 'google/siglip2-so400m-patch16-naflex',
                status TEXT DEFAULT 'completed'
            );
            CREATE TABLE scene_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER NOT NULL,
                scene_index INTEGER NOT NULL,
                start_time_ms INTEGER NOT NULL,
                end_time_ms INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                thumbnail BLOB,
                FOREIGN KEY (video_id) REFERENCES processed_videos(id) ON DELETE CASCADE
            );
            CREATE TABLE scene_motion (
                scene_id INTEGER PRIMARY KEY,
                motion_version INTEGER NOT NULL,
                start_raw TEXT,
                end_raw TEXT,
                FOREIGN KEY (scene_id) REFERENCES scene_embeddings(id) ON DELETE CASCADE
            );
            PRAGMA user_version = 3;
            """
        )
        conn.execute(
            "INSERT INTO image_embeddings VALUES (?, ?, ?, ?, ?)",
            ("D:/Anime/Frieren/poster.jpg", 1.0, b"\x00" * 32, "google/siglip2-so400m-patch16-naflex", "image"),
        )
        conn.execute(
            "INSERT INTO processed_videos (id, filepath, modified_at, model_version, status) VALUES (1, ?, 100.0, ?, 'completed')",
            ("D:/Anime/Frieren/Season 1/Ep01.mkv", "google/siglip2-so400m-patch16-naflex"),
        )
        conn.execute(
            "INSERT INTO scene_embeddings (id, video_id, scene_index, start_time_ms, end_time_ms, embedding, thumbnail) VALUES (10, 1, 0, 0, 2000, ?, ?)",
            (q.tobytes(), jpg_bytes),
        )
        conn.execute(
            "INSERT INTO scene_embeddings (id, video_id, scene_index, start_time_ms, end_time_ms, embedding, thumbnail) VALUES (11, 1, 1, 2000, 4000, ?, NULL)",
            (v2.tobytes(),),
        )
        conn.execute(
            "INSERT INTO scene_motion (scene_id, motion_version, start_raw, end_raw) VALUES (10, ?, ?, NULL)",
            (
                motion.MOTION_VERSION,
                '{"v":2,"pairs":6,"cam":[0.0,0.0,0.0],"camConf":0.9,"subjPairs":6,"subjSupport":40,"subj":[0.5,0.0],"subjRel":[0.5,0.0],"subjConsistency":0.95}',
            ),
        )
        conn.commit()

    info_ro = scoutdb.database_info(db_file)
    assert info_ro.schema_status == "needs_upgrade"
    assert info_ro.schema_version == 3
    assert info_ro.video_count == 1
    assert info_ro.scene_count == 2

    # Must NOT silently mutate v3 on open_database or search
    with pytest.raises(SchemaUpgradeRequiredError):
        scoutdb.open_database(db_file)
    with pytest.raises(SchemaUpgradeRequiredError):
        search(q, SearchOptions(databases=[str(db_file)], include_thumbnails=False))
    with sqlite3.connect(str(db_file)) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 3

    info, bak_path = scoutdb.upgrade_database(db_file, backup=True)
    assert bak_path is not None and bak_path.exists()
    assert bak_path.name == "legacy_v3.scoutdb.v3.bak"
    assert info.schema_status == "ready"
    assert info.schema_version == 4
    assert info.created_by_app == "unknown"
    assert info.video_count == 1
    assert info.scene_count == 2
    assert "D:/Anime/Frieren/Season 1" in info.media_roots

    # Backup itself remains a valid v3 database
    with sqlite3.connect(str(bak_path)) as bconn:
        assert bconn.execute("PRAGMA user_version").fetchone()[0] == 3
        assert bconn.execute("SELECT COUNT(*) FROM scene_embeddings").fetchone()[0] == 2

    hits = search(q, SearchOptions(databases=[str(db_file)], include_thumbnails=True))
    assert len(hits) == 2
    assert hits[0].scene_index == 0
    assert hits[0].thumbnail_b64 is not None
    assert hits[0].motion["start"]["subject"] == "right"

    # Deleting the video with foreign_keys=ON must NOT fail with "no such table: main.scene_embeddings"
    assert scoutdb.remove_video(db_file, video_id=1) is True
    assert scoutdb.database_info(db_file).scene_count == 0


def test_content_fingerprint_small_and_large_files(tmp_path: Path):
    small = tmp_path / "small.mp4"
    small.write_bytes(b"hello-scene-scout" * 100)
    sz_s, fp_s = compute_content_fingerprint(small)
    assert sz_s == small.stat().st_size
    assert fp_s.startswith(f"{FINGERPRINT_ALGO}:")
    assert verify_content_fingerprint(small, sz_s, fp_s)

    # > 3 MiB file exercises head + middle + tail sampling
    large = tmp_path / "large.mkv"
    blob = bytearray(4 * 1_048_576 + 12345)
    blob[:16] = b"HEAD_SIGNATURE!!"
    mid = (len(blob) - 1_048_576) // 2
    blob[mid : mid + 16] = b"MID__SIGNATURE!!"
    blob[-16:] = b"TAIL_SIGNATURE!!"
    large.write_bytes(blob)

    sz_l, fp_l = compute_content_fingerprint(large)
    assert verify_content_fingerprint(large, sz_l, fp_l)

    # Tamper 1 byte in tail -> fingerprint must reject
    blob[-1] ^= 0xFF
    large.write_bytes(blob)
    assert not verify_content_fingerprint(large, sz_l, fp_l)


def test_media_root_consolidation_and_smart_relink(tmp_path: Path):
    old_drive = tmp_path / "old_drive" / "Anime" / "Bocchi"
    (old_drive / "Season 1").mkdir(parents=True)
    (old_drive / "Season 2").mkdir(parents=True)

    ep1 = old_drive / "Season 1" / "01.mkv"
    ep2 = old_drive / "Season 2" / "01.mkv"  # identical basename in different subfolder!
    ep3 = old_drive / "Season 2" / "special_raw.mkv"
    ep1.write_bytes(b"EPISODE_ONE_DATA_" * 500)
    ep2.write_bytes(b"EPISODE_TWO_DATA_" * 600)
    ep3.write_bytes(b"EPISODE_THREE_UNIQUE_PAYLOAD_" * 700)

    db_file = scoutdb.create_database("bocchi", root=tmp_path)
    for ep in (ep1, ep2, ep3):
        f_sz, f_fp = compute_content_fingerprint(ep)
        vid = scoutdb.upsert_video(
            db_file,
            str(ep),
            ep.stat().st_mtime,
            status="completed",
            detector="transnetv2",
            sampling_strategy="first_or_midpoint_v1",
            sampling_config={"frames_per_scene": 1, "pooling": "none", "max_patches": 256},
            file_size=f_sz,
            content_fingerprint=f_fp,
            duration_ms=1440_000,
            fps=23.976,
            width=1920,
            height=1080,
        )
        scoutdb.insert_video_scenes(db_file, vid, [(0, 2000)], _unit_vec(vid).reshape(1, -1))

    info = scoutdb.database_info(db_file)
    # Consolidated Season 1 + Season 2 into common parent .../Anime/Bocchi
    assert len(info.media_roots) == 1
    assert info.media_roots[0].endswith("/Anime/Bocchi")

    # Simulate moving folder to a new drive AND renaming ep3!
    new_drive = tmp_path / "new_drive" / "Library" / "Bocchi"
    (new_drive / "Season 1").mkdir(parents=True)
    (new_drive / "Season 2").mkdir(parents=True)
    new_ep1 = new_drive / "Season 1" / "01.mkv"
    new_ep2 = new_drive / "Season 2" / "01.mkv"
    new_ep3 = new_drive / "Season 2" / "Bocchi_OVA_Renamed.mkv"
    ep1.rename(new_ep1)
    ep2.rename(new_ep2)
    ep3.rename(new_ep3)

    summary = scoutdb.relink_database_paths(db_file, [new_drive])
    assert summary["missing"] == 3
    assert summary["relinked"] == 3
    assert summary["unresolved"] == 0

    videos = {v.id: Path(v.filepath) for v in scoutdb.list_videos(db_file)}
    assert set(videos.values()) == {new_ep1.resolve(), new_ep2.resolve(), new_ep3.resolve()}


def test_exclude_video_paths_and_lru_cache_lifecycle(tmp_path: Path):
    EMBEDDING_CACHE.clear()
    assert EMBEDDING_CACHE.used_bytes == 0

    db_file = scoutdb.create_database("cache_test", root=tmp_path)
    q = _unit_vec(42)
    for idx, name in enumerate(("ep1.mkv", "ep2.mkv")):
        vid = scoutdb.upsert_video(db_file, f"D:/Show/{name}", 1.0, status="completed", duration_ms=10_000)
        vec = q if idx == 0 else (q * 0.8 + _unit_vec(99) * 0.2)
        vec = vec / np.linalg.norm(vec)
        scoutdb.insert_video_scenes(db_file, vid, [(0, 5000)], vec.reshape(1, -1))

    hits_all = search(q, SearchOptions(databases=[str(db_file)], include_thumbnails=False))
    assert len(hits_all) == 2
    assert EMBEDDING_CACHE.used_bytes > 0

    hits_excl = search(
        q,
        SearchOptions(
            databases=[str(db_file)],
            exclude_video_paths=["D:/Show/ep1.mkv"],
            include_thumbnails=False,
        ),
    )
    assert [h.video_path for h in hits_excl] == ["D:/Show/ep2.mkv"]

    # Standby / unload clears embedding cache back to 0 bytes
    scout_embedding.unload_model()
    assert EMBEDDING_CACHE.used_bytes == 0

    # Small cap cache evicts LRU entry
    tiny_cache = EmbeddingCache(max_bytes=64)
    p1 = scoutdb.create_database("c1", root=tmp_path)
    p2 = scoutdb.create_database("c2", root=tmp_path)
    for p in (p1, p2):
        v = scoutdb.upsert_video(p, "D:/Show/a.mkv", 1.0, status="completed")
        scoutdb.insert_video_scenes(p, v, [(0, 1000)], np.ones((1, 8), dtype=np.float32))  # 32 bytes each
    from amverge.core.scenescout.search import _load_db_videos

    tiny_cache.get_or_load(p1, _load_db_videos)
    tiny_cache.get_or_load(p2, _load_db_videos)
    assert tiny_cache.used_bytes == 64
    p3 = scoutdb.create_database("c3", root=tmp_path)
    v3 = scoutdb.upsert_video(p3, "D:/Show/b.mkv", 1.0, status="completed")
    scoutdb.insert_video_scenes(p3, v3, [(0, 1000)], np.ones((1, 8), dtype=np.float32))
    tiny_cache.get_or_load(p3, _load_db_videos)
    assert tiny_cache.used_bytes <= 64
    assert tiny_cache.entry_count == 2


def test_webp_thumbnail_encode_and_fast_width_detection():
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[:, :320] = [50, 120, 220]
    webp_blob = indexing.encode_thumbnail(frame)
    assert webp_blob[:4] == b"RIFF" and webp_blob[8:12] == b"WEBP"
    assert indexing._fast_image_width(webp_blob) == indexing.THUMBNAIL_WIDTH
    assert not indexing.thumbnail_needs_redo(webp_blob)

    small_webp = io.BytesIO()
    Image.new("RGB", (160, 90)).save(small_webp, format="WEBP", quality=75)
    assert indexing._fast_image_width(small_webp.getvalue()) == 160
    assert indexing.thumbnail_needs_redo(small_webp.getvalue())


def test_unsupported_newer_schema_reports_metadata_and_polluted_v4_self_heals(tmp_path: Path):
    # 1. Newer schema (v5) reports updated_by_app/version read-only without mutating user_version
    db_v5 = tmp_path / "future_v5.scoutdb"
    with sqlite3.connect(str(db_v5)) as conn:
        conn.executescript(
            """
            CREATE TABLE db_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE processed_videos (id INTEGER PRIMARY KEY, filepath TEXT UNIQUE NOT NULL, modified_at REAL NOT NULL, scene_count INTEGER DEFAULT 0);
            INSERT INTO db_metadata (key, value) VALUES ('schema_version', '5'), ('updated_by_app', 'amverge'), ('updated_by_version', '9.9.0');
            PRAGMA user_version = 5;
            """
        )
        conn.commit()

    info_v5 = scoutdb.database_info(db_v5)
    assert info_v5.schema_status == "unsupported_newer"
    assert info_v5.schema_version == 5
    assert info_v5.updated_by_app == "amverge"
    assert info_v5.updated_by_version == "9.9.0"

    with pytest.raises(UnsupportedSchemaError, match="amverge 9.9.0"):
        scoutdb.open_database(db_v5)
    with pytest.raises(UnsupportedSchemaError, match="v5"):
        scoutdb.upgrade_database(db_v5)

    # 2. Polluted v4 DB (if opened by a pre-v4 app that added empty v3 tables & reset user_version=3)
    db_polluted = scoutdb.create_database("polluted", root=tmp_path)
    q = _unit_vec(77)
    vid = scoutdb.upsert_video(db_polluted, "D:/Anime/Ep01.mkv", 1.0, status="completed")
    scoutdb.insert_video_scenes(db_polluted, vid, [(0, 2000)], q.reshape(1, -1))

    with sqlite3.connect(str(db_polluted)) as conn:
        conn.executescript(
            """
            CREATE TABLE scene_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id INTEGER NOT NULL,
                scene_index INTEGER NOT NULL,
                start_time_ms INTEGER NOT NULL,
                end_time_ms INTEGER NOT NULL,
                embedding BLOB NOT NULL,
                thumbnail BLOB
            );
            CREATE TABLE image_embeddings (
                filepath TEXT PRIMARY KEY,
                modified_at REAL NOT NULL,
                embedding BLOB NOT NULL
            );
            PRAGMA user_version = 3;
            """
        )
        conn.commit()

    healed = scoutdb.open_database(db_polluted)
    assert healed.schema_status == "ready"
    assert healed.schema_version == 4
    assert healed.video_count == 1
    assert healed.scene_count == 1

    hits = search(q, SearchOptions(databases=[str(db_polluted)], include_thumbnails=False))
    assert len(hits) == 1
    with sqlite3.connect(str(db_polluted)) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 4
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert "scene_embeddings" not in tables
        assert "image_embeddings" not in tables


def test_compact_reclaims_freelist_even_when_external_wal_holder_is_open(tmp_path: Path):
    db_file = tmp_path / "wal_locked_v3.scoutdb"
    vec = _unit_vec(5, dim=64).tobytes() * 200
    with sqlite3.connect(str(db_file)) as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(
            """
            CREATE TABLE processed_videos (id INTEGER PRIMARY KEY AUTOINCREMENT, filepath TEXT UNIQUE NOT NULL, modified_at REAL NOT NULL, model_version TEXT, status TEXT DEFAULT 'completed');
            CREATE TABLE scene_embeddings (id INTEGER PRIMARY KEY AUTOINCREMENT, video_id INTEGER NOT NULL, scene_index INTEGER NOT NULL, start_time_ms INTEGER NOT NULL, end_time_ms INTEGER NOT NULL, embedding BLOB NOT NULL, thumbnail BLOB);
            PRAGMA user_version = 3;
            """
        )
        conn.execute("INSERT INTO processed_videos (id, filepath, modified_at) VALUES (1, 'D:/Show/ep1.mkv', 1.0)")
        conn.executemany(
            "INSERT INTO scene_embeddings (video_id, scene_index, start_time_ms, end_time_ms, embedding, thumbnail) VALUES (1, ?, 0, 1000, ?, NULL)",
            [(i, vec) for i in range(40)],
        )
        conn.commit()

    # Simulate another app (e.g. old Scene Scout) holding an idle WAL connection open
    holder = sqlite3.connect(str(db_file))
    holder.execute("PRAGMA journal_mode = WAL")
    holder.execute("SELECT COUNT(*) FROM processed_videos").fetchone()
    try:
        scoutdb.upgrade_database(db_file, backup=False)
    finally:
        holder.close()

    with sqlite3.connect(str(db_file)) as conn:
        assert conn.execute("PRAGMA freelist_count").fetchone()[0] == 0
        assert conn.execute("PRAGMA auto_vacuum").fetchone()[0] == 2


def test_wal_write_invalidates_embedding_cache_cross_process(tmp_path: Path):
    cache = EmbeddingCache(max_bytes=1024 * 1024)
    db_file = scoutdb.create_database("wal_cache", root=tmp_path)
    q = _unit_vec(88)
    v1 = scoutdb.upsert_video(db_file, "D:/Show/ep1.mkv", 1.0, status="completed")
    scoutdb.insert_video_scenes(db_file, v1, [(0, 1000)], q.reshape(1, -1))

    from amverge.core.scenescout.search import _load_db_videos

    first = cache.get_or_load(db_file, _load_db_videos)
    assert len(first) == 1

    # Hold a WAL read connection so the next write lands in -wal without checkpointing the main DB
    ro_holder = sqlite3.connect(str(db_file))
    ro_holder.execute("SELECT COUNT(*) FROM processed_videos").fetchone()
    try:
        v2 = scoutdb.upsert_video(db_file, "D:/Show/ep2.mkv", 2.0, status="completed")
        scoutdb.insert_video_scenes(db_file, v2, [(0, 1000)], q.reshape(1, -1))
        cache_no_local_inval = EmbeddingCache(max_bytes=1024 * 1024)
        # Seed a cache with the old entry's stat_sig before -wal changed, or directly test stat_sig mismatch:
        wal = db_file.with_name(f"{db_file.name}-wal")
        assert wal.exists() and wal.stat().st_size > 0
        second = cache.get_or_load(db_file, _load_db_videos)
        assert len(second) == 2
        del cache_no_local_inval
    finally:
        ro_holder.close()


def test_v1_to_v4_atomic_migration_and_fk_integrity(tmp_path: Path):
    db_file = tmp_path / "legacy_v1.scoutdb"
    q = _unit_vec(12)
    with sqlite3.connect(str(db_file)) as conn:
        conn.executescript(
            """
            CREATE TABLE processed_videos (
                filepath TEXT UNIQUE NOT NULL,
                modified_at REAL NOT NULL,
                model_version TEXT DEFAULT 'google/siglip2-so400m-patch16-naflex'
            );
            CREATE TABLE scene_embeddings (
                filepath TEXT NOT NULL,
                scene_index INTEGER NOT NULL,
                start_time_ms INTEGER NOT NULL,
                end_time_ms INTEGER NOT NULL,
                embedding BLOB NOT NULL
            );
            PRAGMA user_version = 1;
            """
        )
        conn.execute("INSERT INTO processed_videos VALUES ('D:/Anime/Old/ep1.mp4', 5.0, 'google/siglip2-so400m-patch16-naflex')")
        conn.execute("INSERT INTO scene_embeddings VALUES ('D:/Anime/Old/ep1.mp4', 0, 0, 2000, ?)", (q.tobytes(),))
        conn.commit()

    info, bak = scoutdb.upgrade_database(db_file, backup=True)
    assert info.schema_version == 4
    assert info.video_count == 1
    assert info.scene_count == 1
    assert bak is not None and bak.exists()
    with sqlite3.connect(str(db_file)) as conn:
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
