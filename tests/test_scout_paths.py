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
