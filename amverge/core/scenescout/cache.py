"""Bounded, on-demand LRU cache of per-video Scene Scout embedding matrices.

Starts at 0 bytes, caches only databases that are actively searched up to ``max_bytes``
(default 512 MiB), invalidates automatically whenever a database file's ``mtime_ns`` or
byte size changes on disk, and clears back to 0 bytes on idle standby or model unload.
"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

DEFAULT_MAX_CACHE_BYTES = 512 * 1024 * 1024  # 512 MiB ceiling (0 bytes reserved initially)


@dataclass(frozen=True)
class CachedVideoEmbeddings:
    video_id: int
    filepath: str
    norm_filepath: str
    duration_ms: Optional[int]
    scene_count: int
    dim: int
    dtype: str
    blob: bytes


@dataclass
class _CacheEntry:
    stat_sig: tuple[int, int, int, int]
    byte_cost: int
    videos: list[CachedVideoEmbeddings]


def _db_stat_signature(resolved: Path) -> Optional[tuple[int, int, int, int]]:
    """Return ``(db_mtime_ns, db_size, wal_mtime_ns, wal_size)`` so WAL writes invalidate too."""
    try:
        st = os.stat(resolved)
    except OSError:
        return None
    wal = resolved.with_name(f"{resolved.name}-wal")
    try:
        wst = os.stat(wal)
        wal_sig = (wst.st_mtime_ns, wst.st_size)
    except OSError:
        wal_sig = (0, 0)
    return (st.st_mtime_ns, st.st_size, wal_sig[0], wal_sig[1])


class EmbeddingCache:
    """Thread-safe LRU cache keyed by resolved database path."""

    def __init__(self, max_bytes: int = DEFAULT_MAX_CACHE_BYTES) -> None:
        self.max_bytes = max(0, int(max_bytes))
        self._lock = threading.RLock()
        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._used_bytes = 0

    @property
    def used_bytes(self) -> int:
        with self._lock:
            return self._used_bytes

    @property
    def entry_count(self) -> int:
        with self._lock:
            return len(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._used_bytes = 0

    def invalidate(self, db_path: str | Path) -> None:
        key = str(Path(db_path).resolve())
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self._used_bytes = max(0, self._used_bytes - old.byte_cost)

    def get_or_load(
        self,
        db_path: str | Path,
        loader: Callable[[Path], list[CachedVideoEmbeddings]],
    ) -> list[CachedVideoEmbeddings]:
        resolved = Path(db_path).resolve()
        key = str(resolved)
        sig = _db_stat_signature(resolved)
        if sig is None:
            self.invalidate(resolved)
            return []

        with self._lock:
            cached = self._entries.get(key)
            if cached is not None and cached.stat_sig == sig:
                self._entries.move_to_end(key)
                return cached.videos
            if cached is not None:
                self._entries.pop(key, None)
                self._used_bytes = max(0, self._used_bytes - cached.byte_cost)

        videos = loader(resolved)
        byte_cost = sum(len(v.blob) for v in videos)

        if self.max_bytes <= 0 or byte_cost > self.max_bytes:
            return videos

        latest_sig = _db_stat_signature(resolved) or sig
        with self._lock:
            existing = self._entries.pop(key, None)
            if existing is not None:
                self._used_bytes = max(0, self._used_bytes - existing.byte_cost)
            while self._entries and self._used_bytes + byte_cost > self.max_bytes:
                _, evicted = self._entries.popitem(last=False)
                self._used_bytes = max(0, self._used_bytes - evicted.byte_cost)
            self._entries[key] = _CacheEntry(
                stat_sig=latest_sig,
                byte_cost=byte_cost,
                videos=videos,
            )
            self._used_bytes += byte_cost
        return videos


EMBEDDING_CACHE = EmbeddingCache()


def clear_embedding_cache() -> None:
    EMBEDDING_CACHE.clear()


def invalidate_embedding_cache(db_path: str | Path) -> None:
    EMBEDDING_CACHE.invalidate(db_path)
