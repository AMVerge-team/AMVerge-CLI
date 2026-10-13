"""Which stored thumbnails `scout generate-thumbnails` regenerates."""

import io

from PIL import Image

from amverge.core.scenescout.indexing import THUMBNAIL_WIDTH, thumbnail_needs_redo


def jpeg(width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height)).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_missing_thumbnail_is_generated():
    assert thumbnail_needs_redo(None)


def test_old_low_res_thumbnail_is_regenerated():
    # what scene-scout-dev databases store
    assert thumbnail_needs_redo(jpeg(160, 90))


def test_current_size_thumbnail_is_kept():
    assert not thumbnail_needs_redo(jpeg(THUMBNAIL_WIDTH, 180))


def test_unreadable_blob_is_regenerated():
    assert thumbnail_needs_redo(b"not a jpeg")
