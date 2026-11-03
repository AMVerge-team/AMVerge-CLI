"""Shared value types for Scene Scout.

Kept free of any torch / transformers import so the CLI can list databases and
report status on a machine that has never installed the model.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Optional


# bump when the model or preprocessing changes, so older databases are re-indexed, not compared across spaces
EMBEDDING_MODEL_VERSION = "google/siglip2-so400m-patch16-naflex"


@dataclass(frozen=True)
class SceneHit:
    """One search result: a time range inside an indexed video."""

    video_path: str
    scene_index: int
    start_ms: int
    end_ms: int
    score: float
    database: str
    # base64 jpeg, only when thumbnails are asked for
    thumbnail_b64: Optional[str] = None
    # {"start": tags, "end": tags}; None when indexed without motion
    motion: Optional[dict[str, Any]] = None

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
            "motion": self.motion,
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
    # -1 disables the cut ("show everything" in the app)
    similarity_threshold: float = -1.0
    include_thumbnails: bool = True
    max_patches: int = 256
    batch_size: int = 16
    device: Optional[str] = None
    databases: list[str] = field(default_factory=list)
    video_paths: list[str] = field(default_factory=list)
    # keep only scenes whose start subject moves one of these ways (motion.SUBJECT_TAGS)
    start_subjects: list[str] = field(default_factory=list)
    # whether a scene with no readable motion survives the start_subjects filter
    include_unclear: bool = False
    # (video path, start_ms, end_ms) left out so a flow clip that is itself indexed can't match itself
    exclude_range: Optional[tuple[str, int, int]] = None
    # leaves out scenes this close to their video's start or end; a video too short to have a middle keeps all
    skip_edges_ms: int = 0

    def to_json(self) -> dict[str, Any]:
        return asdict(self)
