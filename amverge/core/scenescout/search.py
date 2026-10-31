"""Similarity search over one or more Scene Scout databases.

The scoring is a dot product over normalised vectors, and the only part that
needs the model is turning the query into a vector, which :mod:`embedding` owns.

Motion tags (see :mod:`motion`) ride along on every hit for display, and can
filter by start direction; they never change the ranking. A flow search ranks
purely by visual similarity to the clip's last frame: in a blind judging round,
scenes ranked by motion direction flowed less often than random ones, while the
most visually similar ones flowed most often.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from . import db as scoutdb
from . import motion as scoutmotion
from .types import SceneHit, SearchOptions


def _norm_path(path: str) -> str:
    return str(Path(path).resolve()).replace("\\", "/").lower()


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


_CHUNK_ROWS = 4096


def _motion_columns(motion_join: bool) -> str:
    return "m.start_raw, m.end_raw, m.scene_id IS NOT NULL" if motion_join else "NULL, NULL, 0"


def _search_one(
    query: np.ndarray,
    database: str | Path,
    options: SearchOptions,
) -> list[SceneHit]:
    """Score every scene in one database against the query.

    Two passes: embeddings are scored in chunks with one matrix product each, and
    only the top hits then read their thumbnails and motion, so a search never
    loads the thumbnail of every scene in the database.
    """
    if options.top_k <= 0:
        return []
    path = Path(database)
    norm_videos = {p.replace("\\", "/").lower() for p in options.video_paths} if options.video_paths else None
    excluded = options.exclude_range
    excluded_path = _norm_path(excluded[0]) if excluded else None
    blob_size = query.shape[0] * 4
    filter_motion = bool(options.start_subjects)

    candidates: list[tuple[float, int, int, int, int, str, Optional[dict[str, Any]]]] = []
    order = 0

    with scoutdb.connect(path) as conn:
        motion_join = scoutdb.has_motion_table(conn)
        join_motion = filter_motion and motion_join
        cursor = conn.execute(
            f"""
            SELECT s.id, s.start_time_ms, s.end_time_ms, s.embedding, v.filepath,
                   {_motion_columns(join_motion)}
            FROM scene_embeddings s
            JOIN processed_videos v ON v.id = s.video_id
            {"LEFT JOIN scene_motion m ON m.scene_id = s.id" if join_motion else ""}
            WHERE v.status = 'completed'
            """
        )

        while rows := cursor.fetchmany(_CHUNK_ROWS):
            kept = []
            for scene_id, start_ms, end_ms, blob, filepath, start_raw, end_raw, has_motion in rows:
                order += 1
                if len(blob) != blob_size:
                    continue
                if norm_videos is not None and filepath.replace("\\", "/").lower() not in norm_videos:
                    continue
                if (
                    excluded
                    and start_ms < excluded[2]
                    and end_ms > excluded[1]
                    and _norm_path(filepath) == excluded_path
                ):
                    continue
                motion = _scene_motion(start_raw, end_raw, bool(has_motion)) if filter_motion else None
                if filter_motion and not _passes_motion(motion, options):
                    continue
                kept.append((order, scene_id, start_ms, end_ms, blob, filepath, motion))
            if not kept:
                continue

            matrix = np.frombuffer(b"".join(row[4] for row in kept), dtype=np.float32).reshape(len(kept), -1)
            scores = matrix @ query
            picks = np.flatnonzero(scores >= options.similarity_threshold) if options.similarity_threshold >= 0 else np.arange(len(kept))
            if len(picks) > options.top_k:
                picks = picks[np.argpartition(-scores[picks], options.top_k - 1)[: options.top_k]]
            for i in picks:
                row_order, scene_id, start_ms, end_ms, _, filepath, motion = kept[i]
                candidates.append((float(scores[i]), row_order, scene_id, start_ms, end_ms, filepath, motion))

        candidates.sort(key=lambda c: (-c[0], c[1]))
        top = candidates[: options.top_k]
        details = _scene_details(conn, [c[2] for c in top], motion_join, options.include_thumbnails)

    hits = []
    for score, _, scene_id, start_ms, end_ms, filepath, motion in top:
        scene_index, thumb, start_raw, end_raw, has_motion = details[scene_id]
        hits.append(
            SceneHit(
                video_path=filepath,
                scene_index=scene_index,
                start_ms=start_ms,
                end_ms=end_ms,
                score=score,
                database=path.stem,
                thumbnail_b64=base64.b64encode(thumb).decode("ascii") if thumb else None,
                motion=motion if filter_motion else _scene_motion(start_raw, end_raw, bool(has_motion)),
            )
        )
    return hits


def _scene_details(
    conn, scene_ids: list[int], motion_join: bool, include_thumbnails: bool
) -> dict[int, tuple[int, Optional[bytes], Optional[str], Optional[str], int]]:
    """scene index, thumbnail and motion for the given scenes, by scene id."""
    if not scene_ids:
        return {}
    marks = ",".join("?" * len(scene_ids))
    rows = conn.execute(
        f"""
        SELECT s.id, s.scene_index, {"s.thumbnail" if include_thumbnails else "NULL"},
               {_motion_columns(motion_join)}
        FROM scene_embeddings s
        {"LEFT JOIN scene_motion m ON m.scene_id = s.id" if motion_join else ""}
        WHERE s.id IN ({marks})
        """,
        scene_ids,
    ).fetchall()
    return {row[0]: row[1:] for row in rows}


def search(query: np.ndarray, options: SearchOptions) -> list[SceneHit]:
    """Search every selected database and merge the results.

    Each database is reduced to its own top_k first, then the merged list is cut
    again, so one large database cannot crowd out a small one before scores are
    compared.
    """
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
    """Scenes that flow out of the end of a clip, most visually similar first.

    The query is the clip's last frame (or the text, when given). The clip's
    own motion tags are measured too, but only returned for display. Returns
    the hits and those tags.
    """
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
