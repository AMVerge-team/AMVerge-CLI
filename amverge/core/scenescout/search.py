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
import heapq
import json
from pathlib import Path
from typing import Any, Optional

import numpy as np

from . import db as scoutdb
from . import motion as scoutmotion
from .embedding import unpack_embedding
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


def _search_one(
    query: np.ndarray,
    database: str | Path,
    options: SearchOptions,
) -> list[SceneHit]:
    """Score every scene in one database against the query.

    Rows are streamed and reduced through a heap rather than materialised: a
    database over a season of episodes holds tens of thousands of embeddings,
    and sorting all of them to take 24 wastes both time and memory.
    """
    path = Path(database)
    name = path.stem
    best: list[tuple[float, int, SceneHit]] = []
    tie = 0

    norm_videos = {p.replace("\\", "/").lower() for p in options.video_paths} if options.video_paths else None
    excluded = options.exclude_range
    excluded_path = _norm_path(excluded[0]) if excluded else None

    with scoutdb.connect(path) as conn:
        # databases indexed before motion existed (or by upstream scene-scout) have no table
        motion_join = scoutdb.has_motion_table(conn)
        cursor = conn.execute(
            f"""
            SELECT s.scene_index, s.start_time_ms, s.end_time_ms, s.embedding,
                   s.thumbnail, v.filepath,
                   {"m.start_raw, m.end_raw, m.scene_id IS NOT NULL" if motion_join else "NULL, NULL, 0"}
            FROM scene_embeddings s
            JOIN processed_videos v ON v.id = s.video_id
            {"LEFT JOIN scene_motion m ON m.scene_id = s.id" if motion_join else ""}
            WHERE v.status = 'completed'
            """
        )

        while True:
            rows = cursor.fetchmany(1000)
            if not rows:
                break

            for scene_index, start_ms, end_ms, blob, thumb, filepath, start_raw, end_raw, has_motion in rows:
                if norm_videos is not None and filepath.replace("\\", "/").lower() not in norm_videos:
                    continue
                if (
                    excluded
                    and start_ms < excluded[2]
                    and end_ms > excluded[1]
                    and _norm_path(filepath) == excluded_path
                ):
                    continue

                # the direction filter is cheaper than the dot product, so it runs first
                motion = _scene_motion(start_raw, end_raw, bool(has_motion))
                if not _passes_motion(motion, options):
                    continue

                vector = unpack_embedding(blob)
                if vector.shape != query.shape:
                    # a row written by a different model version; comparing it
                    # would produce a meaningless score rather than an error
                    continue

                score = float(np.dot(query, vector))
                if options.similarity_threshold >= 0 and score < options.similarity_threshold:
                    continue

                # tie is a monotonic counter so heapq never has to compare two
                # SceneHit objects when scores are equal
                tie += 1
                if len(best) >= options.top_k and score <= best[0][0]:
                    continue

                hit = SceneHit(
                    video_path=filepath,
                    scene_index=scene_index,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    score=score,
                    database=name,
                    thumbnail_b64=(
                        base64.b64encode(thumb).decode("ascii")
                        if options.include_thumbnails and thumb
                        else None
                    ),
                    motion=motion,
                )
                if len(best) < options.top_k:
                    heapq.heappush(best, (score, tie, hit))
                else:
                    heapq.heapreplace(best, (score, tie, hit))

    return [hit for _, _, hit in sorted(best, key=lambda item: item[0], reverse=True)]


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
    # a whole-file clip spans everything, so only an indexed range can collide with itself
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
