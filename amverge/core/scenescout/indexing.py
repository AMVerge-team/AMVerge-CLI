"""Turning a video into rows in a Scene Scout database.

Feel free to create this in however way you want, what I have here
is something to get you started.

Pipeline: detect scene boundaries, sample a representative frame per scene,
embed the frames, write scenes plus thumbnails.

Scene detection uses AMVerge's own detectors rather than PySceneDetect, so a
scene found here lines up with the clips the user already has from importing
that episode. Which detector runs follows the user's Settings choice; see
:func:`detect_scenes`.

Only two functions below need writing: :func:`detect_scenes` and
:func:`sample_frames`. :func:`index_video` already orchestrates them and does
all the database work, so it is worth reading first to see what it expects back.

The data, start to finish::

    detect_scenes(video, method)   -> [(start_ms, end_ms), ...]        N scenes
    sample_frames(video, scenes)   -> [uint8 (H, W, 3) RGB, ...]       N frames
    embed_frames(frames)           -> float32 (N, D), rows normalised  N vectors
    encode_thumbnail(frame)        -> JPEG bytes                       N thumbs
                                      |
                                      v
                              one scene_embeddings row per scene

All four stay the same length and the same order. `index_video` zips them
together, so one dropped frame shifts every scene after it onto the wrong
timestamp, with nothing to catch it.

How it reaches a user in the app:

1. They pick a video with Add Episode on the Scene Scout page.
2. The app runs ``amverge scout add <video> --db <name> --root <dir>
   --detector <their Settings choice> --json``.
3. Each ``on_progress`` call becomes one JSON line, which Rust forwards to the
   UI as a ``scout_progress`` event and the toolbar draws as progress. Call it
   often enough that a long video does not look frozen.
4. When it finishes, the rows written here are what every later search reads.

Nothing in this file talks to the app directly. Its only obligations are to
write correct rows and to report progress.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Callable, Optional

from . import db as scoutdb
from .types import EMBEDDING_MODEL_VERSION

ProgressCallback = Callable[[str, int, int], None]

# Frames are stored as small JPEGs so a database over a whole season stays in
# the tens of MB rather than the hundreds
THUMBNAIL_WIDTH = 320
THUMBNAIL_QUALITY = 80


# the app's Settings values, passed straight through so the CLI never has to
# know how the setting is spelled in the UI
DETECTOR_AI = "transnetv2_gpu"
DETECTOR_KEYFRAME = "keyframe_detection"


def detect_scenes(
    video: str | Path,
    *,
    method: str = DETECTOR_KEYFRAME,
    accurate: bool = False,
) -> list[tuple[int, int]]:
    """Scene boundaries as ``(start_ms, end_ms)`` pairs.

    TODO(scene-scout): call the matching AMVerge detector for each branch.

    Use AMVerge's own detectors, not PySceneDetect. Scene boundaries then match the clips the user already has from importing that episode, so a search
    result points at a scene they can actually see in their grid.

    Which one runs follows the user's Settings choice, which arrives here as
    ``method``:

    * ``transnetv2_gpu`` -> ``amverge.core.detection.ai_scene_detection``,
      entry point ``decode_and_detect_scenes``. Needs the ``ml`` pack.
    * ``keyframe_detection`` -> ``amverge.core.detection.keyframe``, entry point
      ``detect_scenes_by_keyframe``.

    Fall back to keyframe for an unknown value, and for ``transnetv2_gpu`` when
    the ``ml`` pack turns out to be missing. Indexing should still work rather
    than fail, since keyframe detection needs nothing extra.

    Both AMVerge detectors return seconds; this returns milliseconds, because
    that is what the database columns and the upstream schema use.
    """
    raise NotImplementedError(
        "Wire up scene detection: dispatch on `method` to "
        "amverge.core.detection.ai_scene_detection (transnetv2_gpu) or "
        "amverge.core.detection.keyframe (keyframe_detection)"
    )


def sample_frames(video: str | Path, scenes: list[tuple[int, int]]) -> list:
    """One representative frame per scene.

    TODO(scene-scout): port from src/processing.py.

    In:  the video path, and the scene list :func:`detect_scenes` returned.
    Out: a list of exactly ``len(scenes)`` frames, in the same order. Each frame
         is ``np.uint8`` of shape ``(H, W, 3)`` in **RGB** order.

    The length and order both matter: :func:`index_video` zips this against
    ``scenes`` and against the embeddings, so a dropped frame silently shifts
    every later scene onto the wrong timestamp.

    Take the middle frame of each scene, as upstream does. The first frame of a
    scene often lands in the fade out of the previous one.

    Decode with PyAV (``av``), which is already a base dependency. Do not add
    OpenCV for this, and if you use it anyway remember it returns BGR while the
    embedding step expects RGB.
    """
    raise NotImplementedError("Port frame sampling from scene-scout's src/processing.py")


def encode_thumbnail(frame) -> bytes:
    """JPEG bytes for the ``thumbnail`` BLOB column."""
    from PIL import Image

    image = Image.fromarray(frame)
    if image.width > THUMBNAIL_WIDTH:
        height = round(image.height * THUMBNAIL_WIDTH / image.width)
        image = image.resize((THUMBNAIL_WIDTH, height), Image.LANCZOS)

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=THUMBNAIL_QUALITY)
    return buffer.getvalue()


def index_video(
    database: str | Path,
    video: str | Path,
    *,
    device: Optional[str] = None,
    max_patches: int = 256,
    batch_size: int = 16,
    accurate: bool = False,
    method: str = DETECTOR_KEYFRAME,
    generate_thumbnails: bool = True,
    on_progress: Optional[ProgressCallback] = None,
) -> int:
    """Index one video into a database. Returns the number of scenes written.

    The video row is written with ``status='indexing'`` up front and only flipped
    to ``completed`` at the end, so a run interrupted halfway leaves a row that
    is visibly incomplete rather than one that looks finished but has no scenes.
    """
    from .embedding import embed_frames, pack_embedding

    video_path = Path(video).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"No such video: {video_path}")

    def report(stage: str, done: int, total: int) -> None:
        if on_progress:
            on_progress(stage, done, total)

    report("detecting", 0, 1)
    scenes = detect_scenes(video_path, method=method, accurate=accurate)
    if not scenes:
        return 0

    report("sampling", 0, len(scenes))
    frames = sample_frames(video_path, scenes)

    report("embedding", 0, len(frames))
    vectors = embed_frames(
        frames, device=device, max_patches=max_patches, batch_size=batch_size
    )

    video_id = scoutdb.upsert_video(
        database, str(video_path), video_path.stat().st_mtime, status="indexing"
    )

    rows = []
    for index, ((start_ms, end_ms), vector) in enumerate(zip(scenes, vectors)):
        thumbnail = encode_thumbnail(frames[index]) if generate_thumbnails else None
        rows.append((index, start_ms, end_ms, pack_embedding(vector), thumbnail))

    scoutdb.insert_scenes(database, video_id, rows)
    scoutdb.mark_video_complete(database, video_id)
    report("done", len(rows), len(rows))

    return len(rows)


def model_version() -> str:
    return EMBEDDING_MODEL_VERSION
