"""Similarity search over one or more Scene Scout databases.

This file is complete and does not need porting: the scoring is a dot product
over normalised vectors, and the only part that needs the model is turning the
query into a vector, which :mod:`embedding` owns.
"""

from __future__ import annotations

import base64
import heapq
from pathlib import Path

import numpy as np

from . import db as scoutdb
from .embedding import unpack_embedding
from .types import SceneHit, SearchOptions


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

    with scoutdb.connect(path) as conn:
        cursor = conn.execute(
            """
            SELECT s.scene_index, s.start_time_ms, s.end_time_ms, s.embedding,
                   s.thumbnail, v.filepath
            FROM scene_embeddings s
            JOIN processed_videos v ON v.id = s.video_id
            """
        )

        while True:
            rows = cursor.fetchmany(1000)
            if not rows:
                break

            for scene_index, start_ms, end_ms, blob, thumb, filepath in rows:
                vector = unpack_embedding(blob)
                if vector.shape != query.shape:
                    # a row written by a different model version; comparing it
                    # would produce a meaningless score rather than an error
                    continue

                score = float(np.dot(query, vector))
                if options.similarity_threshold >= 0 and score < options.similarity_threshold:
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
                )

                # tie is a monotonic counter so heapq never has to compare two
                # SceneHit objects when scores are equal
                tie += 1
                if len(best) < options.top_k:
                    heapq.heappush(best, (score, tie, hit))
                elif score > best[0][0]:
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
