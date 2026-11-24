from pathlib import Path
import tempfile
import pytest

from amverge.core.scenescout.paths import looks_like_path, db_path, resolve_db_path
from amverge.core.scenescout import db as scoutdb
from amverge.core.scenescout.indexing import generate_missing_thumbnails


def test_looks_like_path_handles_path_and_str():
    assert looks_like_path("test.scoutdb")
    assert looks_like_path(Path("test.scoutdb"))
    assert looks_like_path("folder/my_db")
    assert looks_like_path(Path("folder/my_db"))
    assert not looks_like_path("simple_name")
    assert not looks_like_path(Path("simple_name"))


def test_resolve_db_path_resolution():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        created = scoutdb.create_database("sample", root=root)
        assert created.is_file()

        assert scoutdb.resolve_db_path(created) == created
        assert scoutdb.resolve_db_path(str(created)) == created
        assert scoutdb.resolve_db_path("sample", root=root) == created
        assert resolve_db_path(created) == created


def test_generate_missing_thumbnails_empty_db():
    with tempfile.TemporaryDirectory() as td:
        created = scoutdb.create_database("sample", root=Path(td))
        count = generate_missing_thumbnails(created)
        assert count == 0


def test_remove_video_by_id_and_filepath():
    with tempfile.TemporaryDirectory() as td:
        created = scoutdb.create_database("sample", root=Path(td))
        vid1 = scoutdb.upsert_video(created, "C:/videos/vid1.mp4", 123456.0, "completed")
        scoutdb.insert_scenes(
            created,
            vid1,
            [
                (0, 0, 1000, b"emb0", b"thumb0"),
                (1, 1000, 2000, b"emb1", b"thumb1"),
            ],
        )

        vid2 = scoutdb.upsert_video(created, "C:/videos/vid2.mp4", 123457.0, "completed")
        scoutdb.insert_scenes(
            created,
            vid2,
            [
                (0, 0, 1500, b"emb2", b"thumb2"),
            ],
        )

        info = scoutdb.database_info(created)
        assert info.video_count == 2
        assert info.scene_count == 3

        removed1 = scoutdb.remove_video(created, video_id=vid1)
        assert removed1 is True

        info = scoutdb.database_info(created)
        assert info.video_count == 1
        assert info.scene_count == 1

        with scoutdb.connect(created) as conn:
            scenes = conn.execute("SELECT COUNT(*) FROM scenes WHERE video_id = ?", (vid1,)).fetchone()[0]
            assert scenes == 0

        removed2 = scoutdb.remove_video(created, filepath="C:/videos/vid2.mp4")
        assert removed2 is True

        info = scoutdb.database_info(created)
        assert info.video_count == 0
        assert info.scene_count == 0

        assert scoutdb.remove_video(created, video_id=9999) is False
