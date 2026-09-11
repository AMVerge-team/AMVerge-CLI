"""Shared value types for Scene Scout.

Kept free of any torch / transformers import so the CLI can list databases and
report status on a machine that has never installed the model.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Optional


# Bumped whenever the embedding model or its preprocessing changes. Rows carry
# it so a database built by an older version can be detected and re-indexed
# rather than silently compared against embeddings from a different space.
EMBEDDING_MODEL_VERSION = "siglip2-so400m-patch16-naflex"


@dataclass(frozen=True)
class SceneHit:
    """One search result: a time range inside an indexed video."""

    video_path: str
    scene_index: int
    start_ms: int
    end_ms: int
    score: float
    database: str
    # base64 JPEG, only populated when the caller asks for thumbnails; a search
    # over a large database returns hundreds of rows and the images dwarf
    # everything else in the payload
    thumbnail_b64: Optional[str] = None

    def to_json(self) -> dict[str, Any]:
        return {
            "videoPath": self.video_path,
            "sceneIndex": self.scene_index,
            "startMs": self.start_ms,
            "endMs": self.end_ms,
            "startSec": self.start_ms / 1000.0,
            "endSec": self.end_ms / 1000.0,
            "score": round(self.score, 6),
            "database": self.database,
            "thumbnailB64": self.thumbnail_b64,
        }


@dataclass(frozen=True)
class DatabaseInfo:
    """A Scene Scout database on disk, plus what it holds."""

    name: str
    path: str
    video_count: int = 0
    scene_count: int = 0
    size_bytes: int = 0
    model_version: str = EMBEDDING_MODEL_VERSION
    created_at: Optional[str] = None

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "videoCount": self.video_count,
            "sceneCount": self.scene_count,
            "sizeBytes": self.size_bytes,
            "modelVersion": self.model_version,
            "createdAt": self.created_at,
        }


@dataclass(frozen=True)
class IndexedVideo:
    """A video that has been (or is being) indexed into a database."""

    id: int
    filepath: str
    scene_count: int
    status: str = "completed"
    modified_at: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "filepath": self.filepath,
            "name": Path(self.filepath).name,
            "sceneCount": self.scene_count,
            "status": self.status,
            "modifiedAt": self.modified_at,
        }


@dataclass
class SearchOptions:
    """Everything the app's search settings dropdown can set.

    Defaults match the app's own defaults so a CLI run without flags behaves the
    same as the app with an untouched settings panel.
    """

    top_k: int = 24
    # -1 disables the cut entirely, which is what "show everything" means in the
    # app's settings dropdown
    similarity_threshold: float = -1.0
    include_thumbnails: bool = True
    max_patches: int = 256
    batch_size: int = 16
    device: Optional[str] = None
    databases: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)
