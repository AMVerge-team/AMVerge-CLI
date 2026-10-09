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
    use_ai = accurate or method == DETECTOR_AI
    if use_ai:
        from ..detection.ai_scene_detection import TRANSNET_AVAILABLE, decode_and_detect_scenes

        if TRANSNET_AVAILABLE:
            scenes_secs, _ = decode_and_detect_scenes(video)
            scenes = [(int(s * 1000), int(e * 1000)) for s, e in scenes_secs]
            if scenes:
                return scenes

    from ..detection.keyframe import detect_scenes_by_keyframe

    keyframe_scenes = detect_scenes_by_keyframe(video)
    return [(int(s * 1000), int(e * 1000)) for s, e in keyframe_scenes]


def sample_frames(
    video: str | Path,
    scenes: list[tuple[int, int]],
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> list:
    """One representative frame per scene.

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

    ``on_progress(done, total)`` fires as frames are collected, about once per
    percent, so a caller can show how far the decode has got.
    """
    import av

    if not scenes:
        return []

    total = len(scenes)
    step = max(1, total // 100)
    last_reported = 0

    def report(done: int) -> None:
        nonlocal last_reported
        if on_progress and done != last_reported and (done - last_reported >= step or done == total):
            last_reported = done
            on_progress(done, total)

    targets = [(s + e) // 2 for s, e in scenes]

    container = av.open(str(video), options={"err_detect": "ignore_err"})
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    out: list = []
    target_idx = 0
    # kept raw: converting is the expensive part, and only the end-of-video padding needs it
    last_raw = None

    try:
        for frame in container.decode(stream):
            try:
                current_ms = int(frame.time * 1000)
            except (TypeError, ValueError):
                continue
            last_raw = frame

            # only a frame that reaches the next scene's midpoint is converted to rgb;
            # every other decoded frame is skipped untouched
            if current_ms < targets[target_idx]:
                continue
            try:
                nd = frame.to_ndarray(format="rgb24")
            except (av.FFmpegError, ValueError):
                # an unconvertible frame leaves the target pending for the next one
                continue

            while target_idx < len(targets) and current_ms >= targets[target_idx]:
                out.append(nd)
                target_idx += 1
            report(target_idx)

            if target_idx >= len(targets):
                break
    finally:
        container.close()

    if len(out) < len(scenes) and last_raw is not None:
        try:
            last_frame = last_raw.to_ndarray(format="rgb24")
        except (av.FFmpegError, ValueError):
            last_frame = None
        if last_frame is not None:
            out.extend([last_frame] * (len(scenes) - len(out)))
    report(len(out))

    return out


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
    from ..infra.ipc import log
    from .embedding import embed_frames, is_model_loaded, pack_embedding

    video_path = Path(video).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"No such video: {video_path}")

    def report(stage: str, done: int, total: int) -> None:
        if on_progress:
            on_progress(stage, done, total)

    report("detecting", 0, 1)
    log(f"[diag] scene scout {video_path.name} | detecting scenes using {method}...")
    scenes = detect_scenes(video_path, method=method, accurate=accurate)
    if not scenes:
        log(f"[diag] scene scout {video_path.name} | no scenes detected")
        return 0

    log(f"[diag] scene scout {video_path.name} | detected {len(scenes)} scenes")
    report("sampling", 0, len(scenes))
    frames = sample_frames(video_path, scenes, on_progress=lambda done, total: report("sampling", done, total))
    log(f"[diag] scene scout {video_path.name} | sampled {len(frames)} frames")

    if not is_model_loaded():
        report("loading_model", 0, 1)
        log(f"[diag] scene scout {video_path.name} | loading SigLIP 2 model weights...")

    report("embedding", 0, len(frames))
    last_log_pct = -1

    def on_embed_progress(done: int, total: int) -> None:
        nonlocal last_log_pct
        report("embedding", done, total)
        if total > 0:
            pct = done * 100 // total
            if pct != last_log_pct and (pct % 20 == 0 or done == total):
                last_log_pct = pct
                log(f"[diag] scene scout {video_path.name} | embedding {done}/{total} ({pct}%)")

    vectors = embed_frames(
        frames,
        device=device,
        max_patches=max_patches,
        batch_size=batch_size,
        on_progress=on_embed_progress,
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
    log(f"[diag] scene scout {video_path.name} | indexed {len(rows)} scenes successfully")

    return len(rows)


def model_version() -> str:
    return EMBEDDING_MODEL_VERSION


def generate_missing_thumbnails(
    database: str | Path,
    *,
    on_progress: Optional[ProgressCallback] = None,
) -> int:
    from ..infra.ipc import log

    db_path = scoutdb.resolve_db_path(database)
    if not db_path.is_file():
        raise FileNotFoundError(f"Database does not exist: {db_path}")

    with scoutdb.connect(db_path) as conn:
        video_rows = conn.execute(
            """
            SELECT pv.id, pv.filepath, COUNT(se.id)
            FROM processed_videos pv
            JOIN scene_embeddings se ON pv.id = se.video_id
            WHERE se.thumbnail IS NULL
            GROUP BY pv.id
            """
        ).fetchall()

        total_missing = sum(row[2] for row in video_rows)
        if total_missing == 0:
            return 0

        updated_count = 0

        for video_id, filepath, _ in video_rows:
            v_path = Path(filepath)
            if not v_path.is_file():
                log(f"[diag] scene scout generate-thumbnails | skipping missing video file: {filepath}")
                continue

            scene_rows = conn.execute(
                """
                SELECT id, start_time_ms, end_time_ms
                FROM scene_embeddings
                WHERE video_id = ? AND thumbnail IS NULL
                ORDER BY scene_index
                """,
                (video_id,),
            ).fetchall()

            if not scene_rows:
                continue

            scenes = [(row[1], row[2]) for row in scene_rows]
            try:
                frames = sample_frames(v_path, scenes)
            except Exception as e:
                log(f"[diag] scene scout generate-thumbnails | decode error on {v_path.name}: {e}")
                continue

            batch_updates = []
            for (scene_id, _, _), frame in zip(scene_rows, frames):
                thumb_bytes = encode_thumbnail(frame)
                batch_updates.append((thumb_bytes, scene_id))

            conn.executemany(
                "UPDATE scene_embeddings SET thumbnail = ? WHERE id = ?",
                batch_updates,
            )
            conn.commit()

            updated_count += len(batch_updates)
            if on_progress:
                on_progress("thumbnails", updated_count, total_missing)

        return updated_count
