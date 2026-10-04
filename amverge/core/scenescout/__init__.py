"""Scene Scout: natural-language scene search over indexed video.

Layered so the cheap parts stay importable without the AI extra:

    paths.py      where databases live (app-owned, CLI-told)
    types.py      value types shared with the JSON contract
    db.py         SQLite storage, schema-compatible with the standalone tool
    search.py     similarity search, complete
    embedding.py  SigLIP 2 model, complete
    indexing.py   video -> scenes -> embeddings, complete

Only `embedding` and `indexing` require torch, and both import it lazily, so
listing databases and reporting status work on a machine without it.
"""

from .paths import STORAGE_DIR_NAME, DB_SUFFIX, resolve_root, db_path, list_db_paths, resolve_db_path
from .types import SceneHit, DatabaseInfo, IndexedVideo, SearchOptions, EMBEDDING_MODEL_VERSION

__all__ = [
    "STORAGE_DIR_NAME",
    "DB_SUFFIX",
    "resolve_root",
    "db_path",
    "list_db_paths",
    "resolve_db_path",
    "SceneHit",
    "DatabaseInfo",
    "IndexedVideo",
    "SearchOptions",
    "EMBEDDING_MODEL_VERSION",
]
