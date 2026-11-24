"""Shared value types for Scene Scout.

Kept free of any torch / transformers import so the CLI can list databases and
report status on a machine that has never installed the model.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Literal, Optional


# bump when the model or preprocessing changes, so older databases are re-indexed, not compared across spaces
EMBEDDING_MODEL_VERSION = "google/siglip2-so400m-patch16-naflex"

# shared SQLite PRAGMA user_version across AMVerge-CLI and standalone Scene Scout
SCHEMA_VERSION = 4

SchemaStatus = Literal["ready", "needs_upgrade", "unsupported_newer"]


class SchemaUpgradeRequiredError(ValueError):
    """Raised when a pre-v4 database requires explicit user confirmation to upgrade."""

    def __init__(
        self,
        path: str | Path,
        current_version: int,
        supported_version: int = SCHEMA_VERSION,
    ) -> None:
        self.path = str(path)
        self.current_version = current_version
        self.supported_version = supported_version
        name = Path(path).name
        super().__init__(
            f"Database '{name}' uses Scene Scout schema v{current_version} and needs to be upgraded to "
            f"v{supported_version} before use. Older versions of AMVerge and Scene Scout cannot read v{supported_version} databases."
        )


class UnsupportedSchemaError(ValueError):
    """Raised when a database uses a newer schema version than this build supports."""

    def __init__(
        self,
        path: str | Path,
        schema_version: int,
        *,
        updated_by_app: Optional[str] = None,
        updated_by_version: Optional[str] = None,
        supported_version: int = SCHEMA_VERSION,
    ) -> None:
        self.path = str(path)
        self.schema_version = schema_version
        self.updated_by_app = updated_by_app
        self.updated_by_version = updated_by_version
        self.supported_version = supported_version
        name = Path(path).name
        writer = ""
        if updated_by_app:
            ver = f" {updated_by_version}" if updated_by_version else ""
            writer = f" (last updated by {updated_by_app}{ver})"
        super().__init__(
            f"Database '{name}' uses Scene Scout schema v{schema_version}{writer}, "
            f"which is newer than supported v{supported_version}. Please update AMVerge."
        )


@dataclass(frozen=True)
class SceneHit:
    """One search result: a time range inside an indexed video."""

    video_path: str
    scene_index: int
    start_ms: int
    end_ms: int
    score: float
    database: str
    # base64 webp/jpeg, only when thumbnails are asked for
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
    database_id: Optional[str] = None
    schema_version: int = SCHEMA_VERSION
    created_by_app: Optional[str] = None
    created_by_version: Optional[str] = None
    updated_by_app: Optional[str] = None
    updated_by_version: Optional[str] = None
    updated_at: Optional[str] = None
    media_roots: list[str] = field(default_factory=list)
    schema_status: SchemaStatus = "ready"

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "videoCount": self.video_count,
            "sceneCount": self.scene_count,
            "sizeBytes": self.size_bytes,
            "modelVersion": self.model_version,
            "createdAt": self.created_at,
            "databaseId": self.database_id,
            "schemaVersion": self.schema_version,
            "schemaStatus": self.schema_status,
            "createdByApp": self.created_by_app,
            "createdByVersion": self.created_by_version,
            "updatedByApp": self.updated_by_app,
            "updatedByVersion": self.updated_by_version,
            "updatedAt": self.updated_at,
            "mediaRoots": list(self.media_roots),
        }


@dataclass(frozen=True)
class IndexedVideo:
    """A video that has been (or is being) indexed into a database."""

    id: int
    filepath: str
    scene_count: int
    status: str = "completed"
    modified_at: float = 0.0
    model_id: Optional[str] = None
    detector: Optional[str] = None
    sampling_strategy: Optional[str] = None
    sampling_config: Optional[dict[str, Any]] = None
    file_size: Optional[int] = None
    content_fingerprint: Optional[str] = None
    duration_ms: Optional[int] = None
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "filepath": self.filepath,
            "name": Path(self.filepath).name,
            "sceneCount": self.scene_count,
            "status": self.status,
            "modifiedAt": self.modified_at,
            "modelId": self.model_id,
            "detector": self.detector,
            "samplingStrategy": self.sampling_strategy,
            "samplingConfig": self.sampling_config,
            "fileSize": self.file_size,
            "contentFingerprint": self.content_fingerprint,
            "durationMs": self.duration_ms,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
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
    exclude_video_paths: list[str] = field(default_factory=list)
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
