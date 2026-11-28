"""Similarity search over one or more Scene Scout databases (v4 + v3 fallback).

Scoring runs over contiguous per-video ``float32`` matrices from ``video_embeddings``
(cached on-demand up to 512 MiB in persistent workers via :mod:`cache`), and fetches
timestamps, WebP/JPEG thumbnails, and motion tags only for the final ``top_k`` hits.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from . import db as scoutdb
from . import motion as scoutmotion
from .cache import CachedVideoEmbeddings, EMBEDDING_CACHE
from .types import SceneHit, SearchOptions


def _norm_path(path: str) -> str:
    try:
        return str(Path(path).resolve()).replace("\\", "/").lower()
    except OSError:
        return str(path).replace("\\", "/").lower()


def _norm_raw(path: str) -> str:
    return str(path).replace("\\", "/").lower()


def _scene_motion(start_raw: Optional[str], end_raw: Optional[str], has_row: bool) -> Optional[dict[str, Any]]:
    if not has_row:
        return None
    return scoutmotion.scene_tags(
        json.loads(start_raw) if start_raw else None,
        json.loads(end_raw) if end_raw else None,
    )


_UNMEASURED = {"subject": "unclear", "camera": "unclear", "speed": None}


def _passes_motion(motion: Optional[dict[str, Any]], options: SearchOptions) -> bool:
    """Whether a scene survives the start direction filter (no filter keeps all)."""
    if not options.start_subjects:
        return True
    subject = (motion["start"] if motion else _UNMEASURED)["subject"]
    return subject in options.start_subjects or (options.include_unclear and subject == "unclear")


def _load_db_videos(db_path: Path) -> list[CachedVideoEmbeddings]:
    out: list[CachedVideoEmbeddings] = []
    conn = scoutdb._open_strict_ro(db_path)
    try:
        conn.execute(f"PRAGMA mmap_size={scoutdb.MMAP_SIZE}")
        rows = conn.execute(
            """
            SELECT v.id, v.filepath, v.duration_ms, ve.scene_count, ve.dim, ve.dtype, ve.embeddings
            FROM video_embeddings ve
            JOIN processed_videos v ON v.id = ve.video_id
            WHERE v.status = 'completed'
            ORDER BY v.id ASC
            """
        ).fetchall()
    finally:
        conn.close()
    for vid, filepath, duration_ms, scene_count, dim, dtype, blob in rows:
        if not blob or scene_count <= 0 or dim <= 0:
            continue
        itemsize = 2 if dtype == "float16" else 4
        if len(blob) != int(scene_count) * int(dim) * itemsize:
            continue
        out.append(
            CachedVideoEmbeddings(
                video_id=int(vid),
                filepath=str(filepath),
                norm_filepath=_norm_raw(filepath),
                duration_ms=int(duration_ms) if duration_ms is not None else None,
                scene_count=int(scene_count),
                dim=int(dim),
                dtype=str(dtype),
                blob=bytes(blob),
            )
        )
    return out


def _is_v4_database(path: Path) -> bool:
    try:
        with scoutdb.connect_ro(path) as conn:
            row = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='video_embeddings'"
            ).fetchone()
            return row is not None
    except Exception:
        return False


def _search_one_v4(
    query: np.ndarray,
    path: Path,
    options: SearchOptions,
) -> list[SceneHit]:
    videos = EMBEDDING_CACHE.get_or_load(path, _load_db_videos)
    if not videos:
        return []

    q_dim = int(query.shape[0])
    include_set = {_norm_raw(p) for p in options.video_paths} if options.video_paths else None
    exclude_set = {_norm_raw(p) for p in options.exclude_video_paths} if options.exclude_video_paths else None
    excluded = options.exclude_range
    excluded_norm = _norm_path(excluded[0]) if excluded else None
    filter_motion = bool(options.start_subjects)
    skip_ms = max(0, int(options.skip_edges_ms))

    # Candidate tuple: (score, video_id, scene_index, filepath, precomputed_motion)
    candidates: list[tuple[float, int, int, str, Optional[dict[str, Any]]]] = []

    ro_conn = None
    has_motion = False
    try:
        for ep in videos:
            if ep.dim != q_dim:
                continue
            if include_set is not None and ep.norm_filepath not in include_set:
                continue
            if exclude_set is not None and ep.norm_filepath in exclude_set:
                continue

            ep_excluded = excluded is not None and _norm_path(ep.filepath) == excluded_norm
            needs_scene_scan = skip_ms > 0 or ep_excluded or filter_motion

            keep_mask: Optional[np.ndarray] = None
            motion_map: dict[int, Optional[dict[str, Any]]] = {}

            if needs_scene_scan:
                if ro_conn is None:
                    ro_conn = scoutdb._open_strict_ro(path)
                    ro_conn.execute(f"PRAGMA mmap_size={scoutdb.MMAP_SIZE}")
                    has_motion = scoutdb.has_motion_table(ro_conn)

                join_motion = filter_motion and has_motion
                scene_rows = ro_conn.execute(
                    f"""
                    SELECT s.scene_index, s.start_time_ms, s.end_time_ms,
                           {"m.start_raw, m.end_raw, m.scene_id IS NOT NULL" if join_motion else "NULL, NULL, 0"}
                    FROM scenes s
                    {"LEFT JOIN scene_motion m ON m.scene_id = s.id" if join_motion else ""}
                    WHERE s.video_id = ?
                    ORDER BY s.scene_index ASC
                    """,
                    (ep.video_id,),
                ).fetchall()

                total_ms = ep.duration_ms
                if total_ms is None and scene_rows:
                    total_ms = max(int(r[2]) for r in scene_rows)
                apply_edge_cut = skip_ms > 0 and total_ms is not None and total_ms > 2 * skip_ms

                keep_mask = np.zeros(ep.scene_count, dtype=bool)
                for s_idx, start_ms, end_ms, start_raw, end_raw, has_m in scene_rows:
                    idx = int(s_idx)
                    if idx < 0 or idx >= ep.scene_count:
                        continue
                    if apply_edge_cut and (start_ms < skip_ms or end_ms > total_ms - skip_ms):  # type: ignore[operator]
                        continue
                    if ep_excluded and excluded is not None and start_ms < excluded[2] and end_ms > excluded[1]:
                        continue
                    if filter_motion:
                        m_tags = _scene_motion(start_raw, end_raw, bool(has_m))
                        if not _passes_motion(m_tags, options):
                            continue
                        motion_map[idx] = m_tags
                    keep_mask[idx] = True

                if not np.any(keep_mask):
                    continue

            np_dtype = np.float16 if ep.dtype == "float16" else np.float32
            mat = np.frombuffer(ep.blob, dtype=np_dtype).reshape(ep.scene_count, ep.dim)
            scores = (mat if mat.dtype == np.float32 else mat.astype(np.float32)) @ query

            if keep_mask is not None:
                scores = scores.copy()
                scores[~keep_mask] = -np.inf

            if options.similarity_threshold >= 0:
                picks = np.flatnonzero(scores >= options.similarity_threshold)
            else:
                picks = np.flatnonzero(np.isfinite(scores))

            if picks.size == 0:
                continue
            if len(picks) > options.top_k:
                picks = picks[np.argpartition(-scores[picks], options.top_k - 1)[: options.top_k]]

            for i in picks:
                idx = int(i)
                candidates.append(
                    (float(scores[idx]), ep.video_id, idx, ep.filepath, motion_map.get(idx))
                )

        if not candidates:
            return []

        candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
        top = candidates[: options.top_k]

        if ro_conn is None:
            ro_conn = scoutdb._open_strict_ro(path)
            ro_conn.execute(f"PRAGMA mmap_size={scoutdb.MMAP_SIZE}")
            has_motion = scoutdb.has_motion_table(ro_conn)

        # Batch fetch metadata/thumbnails/motion per video_id instead of top_k separate queries
        by_video: dict[int, list[int]] = {}
        for _score, vid, s_idx, _filepath, _pre_motion in top:
            by_video.setdefault(vid, []).append(s_idx)

        fetched: dict[tuple[int, int], tuple[int, int, Any, Optional[str], Optional[str], int]] = {}
        thumb_col = "t.thumbnail" if options.include_thumbnails else "NULL"
        motion_cols = (
            "m.start_raw, m.end_raw, m.scene_id IS NOT NULL"
            if (has_motion and not filter_motion)
            else "NULL, NULL, 0"
        )
        thumb_join = "LEFT JOIN scene_thumbnails t ON t.scene_id = s.id" if options.include_thumbnails else ""
        motion_join = "LEFT JOIN scene_motion m ON m.scene_id = s.id" if (has_motion and not filter_motion) else ""

        for vid, indices in by_video.items():
            placeholders = ",".join("?" for _ in indices)
            for row in ro_conn.execute(
                f"""
                SELECT s.scene_index, s.start_time_ms, s.end_time_ms, {thumb_col}, {motion_cols}
                FROM scenes s
                {thumb_join}
                {motion_join}
                WHERE s.video_id = ? AND s.scene_index IN ({placeholders})
                """,
                (vid, *indices),
            ):
                fetched[(vid, int(row[0]))] = (int(row[1]), int(row[2]), row[3], row[4], row[5], int(row[6]))

        hits: list[SceneHit] = []
        for score, vid, s_idx, filepath, pre_motion in top:
            row = fetched.get((vid, s_idx))
            if row is None:
                continue
            start_ms, end_ms, thumb, start_raw, end_raw, has_m = row
            hits.append(
                SceneHit(
                    video_path=filepath,
                    scene_index=s_idx,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    score=score,
                    database=path.stem,
                    thumbnail_b64=base64.b64encode(thumb).decode("ascii") if thumb else None,
                    motion=pre_motion if filter_motion else _scene_motion(start_raw, end_raw, bool(has_m)),
                )
            )
        return hits
    finally:
        if ro_conn is not None:
            ro_conn.close()


def _search_one(
    query: np.ndarray,
    database: str | Path,
    options: SearchOptions,
) -> list[SceneHit]:
    """Score every scene in one database against ``query``."""
    if options.top_k <= 0:
        return []
    path = Path(database)
    if not path.is_file():
        return []
    scoutdb.ensure_v4_ready(path)
    return _search_one_v4(query, path, options)


def search(query: np.ndarray, options: SearchOptions) -> list[SceneHit]:
    """Search every selected database and merge the results."""
    hits: list[SceneHit] = []
    for database in options.databases:
        if Path(database).is_file():
            hits.extend(_search_one(query, database, options))

    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[: options.top_k]


def search_text(text: str, options: SearchOptions) -> list[SceneHit]:
    from .embedding import embed_text

    query = embed_text(text, device=options.device, max_patches=options.max_patches)
    return search(query, options)


def search_image(image_path: str, options: SearchOptions) -> list[SceneHit]:
    from .embedding import embed_image

    query = embed_image(image_path, device=options.device, max_patches=options.max_patches)
    return search(query, options)


def search_flow(
    clip_path: str,
    options: SearchOptions,
    text: Optional[str] = None,
    start_ms: Optional[int] = None,
    end_ms: Optional[int] = None,
) -> tuple[list[SceneHit], dict[str, Any]]:
    """Scenes that flow out of the end of a clip, most visually similar first."""
    from .embedding import embed_frames, embed_text

    measured = scoutmotion.measure_clip(clip_path, start_ms, end_ms)
    clip_tags = scoutmotion.scene_tags(measured["start"], measured["end"])
    if end_ms is not None:
        options.exclude_range = (clip_path, start_ms or 0, end_ms)

    if text:
        query = embed_text(text, device=options.device, max_patches=options.max_patches)
    elif measured["lastFrame"] is not None:
        query = embed_frames(
            [measured["lastFrame"]], device=options.device, max_patches=options.max_patches, batch_size=1
        )[0]
    else:
        raise ValueError("Could not read a frame from the clip to search with")
    return search(query, options), clip_tags
