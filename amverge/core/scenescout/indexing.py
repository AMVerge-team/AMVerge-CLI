"""Turning a video into rows in a Scene Scout database.

Feel free to create this in however way you want, what I have here
is something to get you started.

Pipeline: detect scene boundaries, sample a representative frame per scene
without decoding the rest, embed the frames, write scenes plus thumbnails.

Scene detection uses AMVerge's own detectors rather than PySceneDetect, so a
scene found here lines up with the clips the user already has from importing
that episode. Which detector runs follows the user's Settings choice; see
:func:`detect_scenes`.

:func:`index_video` orchestrates these steps and does all the database work, so
it is worth reading first.

The data, start to finish::

    detect_scenes(video, method)       -> [(start_ms, end_ms), ...]        N scenes
    sample_scene_frames(video, scenes) -> [uint8 (H, W, 3) RGB, ...]       N frames
    embed_frames(frames)               -> float32 (N, D), rows normalised  N vectors
    encode_thumbnail(frame)            -> JPEG bytes                       N thumbs
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

import bisect
import io
import queue
import threading
from pathlib import Path
from typing import Callable, Optional

from . import db as scoutdb
from .types import EMBEDDING_MODEL_VERSION

ProgressCallback = Callable[[str, int, int], None]

# small jpegs keep a season's database in the tens of MB
THUMBNAIL_WIDTH = 320
THUMBNAIL_QUALITY = 80

KEYFRAME_SNAP_FRAMES = 2
BLANK_CHECK_WIDTH = 320
BLANK_DETAIL_THRESHOLD = 1.5


# the app's settings values, passed straight through
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


def _sample_size(width: int, height: int, max_patches: int) -> tuple[int, int]:
    """The size a sampled frame is shrunk to: never bigger than the source.

    SigLIP 2 resizes every frame onto a grid of at most ``max_patches`` 16 px
    patches, so a full-resolution frame only costs CPU time and memory. Four times
    the grid's pixels keeps the resize quality, and a thumbnail is never narrower
    than :data:`THUMBNAIL_WIDTH`.
    """
    target_area = max_patches * 16 * 16 * 4
    scale = min(1.0, max((target_area / (width * height)) ** 0.5, THUMBNAIL_WIDTH / width))
    return max(1, round(width * scale)), max(1, round(height * scale))


def sample_frames(
    video: str | Path,
    scenes: list[tuple[int, int]],
    on_progress: Optional[Callable[[int, int], None]] = None,
    max_patches: Optional[int] = None,
) -> list:
    """The middle frame of each scene, decoding the video in full.

    Used for thumbnails, where a midpoint reads better than a scene's opening. Indexing
    uses the much faster :func:`sample_scene_frames` instead.

    In:  the video path, and the scene list :func:`detect_scenes` returned.
    Out: a list of exactly ``len(scenes)`` frames, in the same order. Each frame
         is ``np.uint8`` of shape ``(H, W, 3)`` in **RGB** order.

    Decode with PyAV (``av``), which is already a base dependency. Do not add
    OpenCV for this, and if you use it anyway remember it returns BGR while the
    embedding step expects RGB.

    ``on_progress(done, total)`` fires as frames are collected, about once per
    percent, so a caller can show how far the decode has got.

    ``max_patches`` shrinks each frame to what that SigLIP patch budget can use
    (see :func:`_sample_size`) during the RGB conversion; ``None`` keeps full size.
    """
    import av

    if not scenes:
        return []

    total = len(scenes)
    step = max(1, total // 100) # for progress bar % updates
    last_reported = 0

    def report(done: int) -> None:
        nonlocal last_reported
        if on_progress and done != last_reported and (done - last_reported >= step or done == total):
            last_reported = done
            on_progress(done, total)

    midpoints_ms = [(start_ms + end_ms) // 2 for start_ms, end_ms in scenes]

    from .decode import close_video, open_video

    container = open_video(video)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    def to_rgb(frame):
        if max_patches is None:
            return frame.to_ndarray(format="rgb24")
        # scaled in the same swscale pass as the rgb conversion, so the full-size frame is never built
        width, height = _sample_size(frame.width, frame.height, max_patches)
        return frame.reformat(width=width, height=height, format="rgb24", interpolation="AREA").to_ndarray()

    out: list = []
    next_midpoint = 0
    # kept raw: converting is the expensive part, and only the end-of-video padding needs it
    last_raw = None

    try:
        for frame in container.decode(stream):
            try:
                current_ms = int(frame.time * 1000)
            except (TypeError, ValueError):
                continue
            last_raw = frame

            # only the frame reaching the next scene's midpoint is converted to rgb
            if current_ms < midpoints_ms[next_midpoint]:
                continue
            try:
                nd = to_rgb(frame)
            except (av.FFmpegError, ValueError):
                # an unconvertible frame leaves the midpoint pending for the next one
                continue

            while next_midpoint < len(midpoints_ms) and current_ms >= midpoints_ms[next_midpoint]:
                out.append(nd)
                next_midpoint += 1
            report(next_midpoint)

            if next_midpoint >= len(midpoints_ms):
                break
    finally:
        close_video(container, stream)

    if len(out) < len(scenes) and last_raw is not None:
        try:
            last_frame = to_rgb(last_raw)
        except (av.FFmpegError, ValueError):
            last_frame = None
        if last_frame is not None:
            out.extend([last_frame] * (len(scenes) - len(out)))
    report(len(out))

    return out


def is_blank_frame(frame) -> bool:
    """True for a frame with little to search for: a flash, black, or an empty background.

    Measures fine detail rather than contrast, so a plain two-tone background still
    counts as empty. A false positive only costs one extra decode.
    """
    import numpy as np

    stride = max(1, frame.shape[1] // BLANK_CHECK_WIDTH)
    gray = frame[::stride, ::stride].mean(axis=2)
    detail = (np.abs(np.diff(gray, axis=1)).mean() + np.abs(np.diff(gray, axis=0)).mean()) / 2
    return bool(detail < BLANK_DETAIL_THRESHOLD)


def _packet_times(video: str | Path) -> tuple[list[int], list[int], float]:
    """Keyframe times and all frame times in ms, sorted, plus the frame rate. Reads packets only, no decode."""
    import av

    keyframes: set[int] = set()
    times: set[int] = set()
    with av.open(str(video), options={"err_detect": "ignore_err"}) as container:
        stream = container.streams.video[0]
        rate = float(stream.average_rate or 0) or 24.0
        for packet in container.demux(stream):
            if packet.pts is None:
                continue
            time_ms = _frame_ms(packet.pts, stream.time_base)
            times.add(time_ms)
            if packet.is_keyframe:
                keyframes.add(time_ms)
    return sorted(keyframes), sorted(times), rate


def _frame_ms(pts: int, time_base) -> int:
    return int(float(pts * time_base) * 1000)


def _scene_targets(scenes: list[tuple[int, int]], keyframes: list[int], times: list[int], rate: float) -> list[int]:
    """The exact frame time to sample for each scene: its first frame, or the keyframe it sits on.

    Detectors can be a frame or two off a cut, while an encoder puts its keyframe on
    the cut itself, so a nearby keyframe is the better first frame and costs less.
    """
    tolerance = KEYFRAME_SNAP_FRAMES * 1000 / rate
    targets = []
    for start_ms, _ in scenes:
        at = bisect.bisect_left(keyframes, start_ms)
        near = [k for k in keyframes[max(0, at - 1) : at + 1] if abs(k - start_ms) <= tolerance]
        if near:
            targets.append(min(near, key=lambda k: abs(k - start_ms)))
            continue
        first = bisect.bisect_left(times, start_ms)
        targets.append(times[min(first, len(times) - 1)] if times else start_ms)
    return targets


def sample_scene_frames(
    video: str | Path,
    scenes: list[tuple[int, int]],
    on_progress: Optional[Callable[[int, int], None]] = None,
    max_patches: Optional[int] = None,
    on_frame: Optional[Callable] = None,
) -> list:
    """One representative frame per scene, without decoding the whole video.

    Returns exactly ``len(scenes)`` RGB ``np.uint8`` frames in scene order:
    :func:`index_video` zips them against ``scenes`` and the embeddings, so a dropped
    frame would shift every later scene onto the wrong timestamp. Scenes past the last
    decodable frame get that last frame; a video with no decodable frames raises.

    Each scene gets its first frame, or its midpoint when that first frame is blank
    (see :func:`is_blank_frame`).

    Packets are dropped whenever no wanted frame remains before the next keyframe. A
    wanted keyframe needs only its own packet; any other frame needs every packet up
    to it, including leading pictures decoded after the following keyframe. Empty
    packets flush the decoder's delayed frames, so they are always sent.

    A decoder fed only some packets outputs frames out of order and can omit some, so
    frames are matched to scenes by exact time, never by arrival. Frames that never
    appear and the midpoints of blank scenes are then fetched with a seek each.

    ``on_progress(done, total)`` fires about once per percent. ``max_patches`` is as in
    :func:`sample_frames`. ``on_frame(index, frame)`` sees each frame as it is ready, so
    embedding can start while the decode runs; blank scenes arrive after the rest, so it
    is not in scene order.
    """
    import av

    from .decode import close_video, open_video

    if not scenes:
        return []

    total = len(scenes)
    keyframes, times, rate = _packet_times(video)
    keyframe_set = set(keyframes)
    targets = _scene_targets(scenes, keyframes, times, rate)
    wanted: dict[int, list[int]] = {}
    for index, time_ms in enumerate(targets):
        wanted.setdefault(time_ms, []).append(index)
    between_keyframes = sorted(t for t in wanted if t not in keyframe_set)

    step = max(1, total // 100)
    out: list = [None] * total
    blank: list[int] = []
    done = reported = 0

    def report() -> None:
        nonlocal reported
        if on_progress and done != reported and (done - reported >= step or done == total):
            reported = done
            on_progress(done, total)

    def to_rgb(frame):
        if max_patches is None:
            return frame.to_ndarray(format="rgb24")
        width, height = _sample_size(frame.width, frame.height, max_patches)
        return frame.reformat(width=width, height=height, format="rgb24", interpolation="AREA").to_ndarray()

    def keep(index: int, nd) -> None:
        nonlocal done
        out[index] = nd
        done += 1
        if on_frame is not None:
            on_frame(index, nd)

    def settle(index: int, nd) -> None:
        if is_blank_frame(nd):
            out[index] = nd
            blank.append(index)
        else:
            keep(index, nd)

    def take(frame, time_base) -> None:
        if frame.pts is None:
            return
        indexes = wanted.get(_frame_ms(frame.pts, time_base))
        if not indexes:
            return
        try:
            nd = to_rgb(frame)
        except (av.FFmpegError, ValueError):
            return
        del wanted[_frame_ms(frame.pts, time_base)]
        for index in indexes:
            settle(index, nd)
        report()

    def fetch(time_ms: int):
        try:
            container.seek(time_ms * 1000, backward=True)
            for frame in container.decode(stream):
                if frame.pts is not None and _frame_ms(frame.pts, stream.time_base) >= time_ms:
                    return to_rgb(frame)
        except (av.FFmpegError, ValueError):
            pass
        return None

    container = open_video(video)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    try:
        decoding = True
        next_key = float("inf")
        previous_key = -1
        pending = 0
        last_raw = None

        def between_keyframes_pending() -> bool:
            nonlocal pending
            while pending < len(between_keyframes) and (
                between_keyframes[pending] not in wanted or between_keyframes[pending] < previous_key
            ):
                pending += 1
            return pending < len(between_keyframes) and between_keyframes[pending] < next_key

        for packet in container.demux(stream):
            if not wanted:
                break
            send = decoding
            if packet.is_keyframe and packet.pts is not None:
                key_ms = _frame_ms(packet.pts, stream.time_base)
                at = bisect.bisect_left(keyframes, key_ms)
                previous_key = keyframes[at - 1] if at > 0 else -1
                next_key = keyframes[at + 1] if at + 1 < len(keyframes) else float("inf")
                decoding = between_keyframes_pending()
                send = decoding or key_ms in wanted
            if not send and packet.size:
                continue
            for frame in packet.decode():
                last_raw = frame
                take(frame, stream.time_base)
            decoding = decoding and between_keyframes_pending()

        for time_ms in sorted(wanted):
            nd = fetch(time_ms)
            for index in wanted[time_ms]:
                if nd is not None:
                    settle(index, nd)
        report()

        for index in sorted(blank):
            start_ms, end_ms = scenes[index]
            nd = fetch((start_ms + end_ms) // 2)
            keep(index, nd if nd is not None else out[index])
            report()

        if done < total:
            nd = to_rgb(last_raw) if last_raw is not None else None
            if nd is None:
                raise ValueError(f"No decodable frames in {video}")
            for index in range(total):
                if out[index] is None:
                    keep(index, nd)
            report()
    finally:
        close_video(container, stream)

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


def precut_scenes(raw) -> list[tuple[int, int]]:
    """Scene ranges handed in by the caller instead of detected, as ``(start_ms, end_ms)``.

    The app passes these for an episode it has already imported and cut, read from
    that episode's manifest, so the database's scenes match the clips exactly and
    detection is skipped. Accepts ``[[start_ms, end_ms], ...]``; ranges that are
    not numbers or are empty are dropped, and the rest are sorted by start.
    """
    scenes: list[tuple[int, int]] = []
    for item in raw or []:
        try:
            start_ms, end_ms = int(item[0]), int(item[1])
        except (TypeError, ValueError, IndexError):
            continue
        if start_ms >= 0 and end_ms > start_ms:
            scenes.append((start_ms, end_ms))
    return sorted(scenes)


def load_precut_scenes(path: str | Path) -> list[tuple[int, int]]:
    """:func:`precut_scenes` from a JSON file (``scout add --scenes-json``)."""
    import json

    return precut_scenes(json.loads(Path(path).read_text(encoding="utf-8")))


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
    scenes: Optional[list[tuple[int, int]]] = None,
    on_progress: Optional[ProgressCallback] = None,
) -> int:
    """Index one video into a database. Returns the number of scenes written.

    The video row is written with ``status='indexing'`` up front and only flipped
    to ``completed`` at the end, so a run interrupted halfway leaves a row that
    is visibly incomplete rather than one that looks finished but has no scenes.

    ``scenes`` (from :func:`precut_scenes`) skips detection and indexes exactly
    those ranges, for an episode the app already cut; ``method`` is then unused.

    Motion tags need every scene's last frames, so they are not measured here;
    ``amverge scout motion`` adds them afterward (see :func:`backfill_motion`).
    """
    from ..infra.ipc import log
    from .embedding import is_model_loaded, pack_embedding

    video_path = Path(video).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"No such video: {video_path}")

    def report(stage: str, done: int, total: int) -> None:
        if on_progress:
            on_progress(stage, done, total)

    if scenes:
        log(f"[diag] scene scout {video_path.name} | using {len(scenes)} pre-cut scenes, detection skipped")
    else:
        report("detecting", 0, 1)
        log(f"[diag] scene scout {video_path.name} | detecting scenes using {method}...")
        scenes = detect_scenes(video_path, method=method, accurate=accurate)
        if not scenes:
            log(f"[diag] scene scout {video_path.name} | no scenes detected")
            return 0
        log(f"[diag] scene scout {video_path.name} | detected {len(scenes)} scenes")
    # embeds batches as they are sampled, so the gpu works while the decode runs
    embedder = _BackgroundEmbedder(device=device, max_patches=max_patches, batch_size=batch_size)
    report("sampling", 0, len(scenes))
    try:
        frames = sample_scene_frames(
            video_path,
            scenes,
            on_progress=lambda done, total: report("sampling", done, total),
            max_patches=max_patches,
            on_frame=embedder.add,
        )
    except BaseException:
        embedder.abandon()
        raise
    log(f"[diag] scene scout {video_path.name} | sampled {len(frames)} frames")

    # announced only now: mid-decode it would take the sampling bar over
    if not is_model_loaded():
        report("loading_model", 0, 1)
        log(f"[diag] scene scout {video_path.name} | loading SigLIP 2 model weights...")

    last_log_pct = -1

    def on_embed_progress(done: int, total: int) -> None:
        nonlocal last_log_pct
        report("embedding", done, total)
        if total > 0:
            pct = done * 100 // total
            if pct != last_log_pct and (pct % 20 == 0 or done == total):
                last_log_pct = pct
                log(f"[diag] scene scout {video_path.name} | embedding {done}/{total} ({pct}%)")

    vectors = embedder.finish(len(frames), on_embed_progress)

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


_END = object()


class _BackgroundEmbedder:
    """Embeds sampled frames on a worker thread, a batch at a time, while the decode runs.

    Frames may arrive in any order; :meth:`finish` returns their vectors in scene
    order. The model loads on the worker too, so that wait overlaps the decode as well.
    """

    def __init__(self, *, device: Optional[str], max_patches: int, batch_size: int):
        self._queue: queue.Queue = queue.Queue()
        self._rows: list = []
        self._order: list[int] = []
        self._done = 0
        self._error: Optional[BaseException] = None
        self._abandoned = False
        self._thread = threading.Thread(target=self._run, args=(device, max_patches, batch_size), daemon=True)
        self._thread.start()

    def add(self, index: int, frame) -> None:
        self._queue.put((index, frame))

    def _run(self, device: Optional[str], max_patches: int, batch_size: int) -> None:
        from .embedding import embed_frames

        batch: list = []
        try:
            while True:
                item = self._queue.get()
                if self._abandoned:
                    return
                if item is not _END:
                    self._order.append(item[0])
                    batch.append(item[1])
                if batch and (item is _END or len(batch) >= batch_size):
                    self._rows.append(embed_frames(batch, device=device, max_patches=max_patches, batch_size=batch_size))
                    self._done += len(batch)
                    batch = []
                if item is _END:
                    return
        except BaseException as exc:  # noqa: BLE001 - handed to the caller in finish()
            self._error = exc

    def finish(self, total: int, on_progress: Callable[[int, int], None]):
        """Waits for the rest, reporting progress, and returns the ``(N, D)`` vectors."""
        import numpy as np

        self._queue.put(_END)
        reported = -1
        while self._thread.is_alive():
            self._thread.join(timeout=0.2)
            if self._done != reported:
                reported = self._done
                on_progress(reported, total)
        if self._error is not None:
            raise self._error
        if reported != self._done:
            on_progress(self._done, total)
        if not self._rows:
            return np.empty((0, 0), dtype=np.float32)
        return np.vstack(self._rows)[np.argsort(self._order)]

    def abandon(self) -> None:
        """Stops the worker after a failed decode: frames still queued are dropped, not embedded."""
        self._abandoned = True
        self._queue.put(_END)


def backfill_motion(
    database: str | Path,
    on_progress: Optional[ProgressCallback] = None,
) -> int:
    """Measures motion for every indexed video that lacks it. Returns scenes written.

    Needs no model, only the source videos still on disk; a moved or deleted
    source is skipped with a log line.
    """
    from ..infra.ipc import log
    from .motion import MOTION_VERSION, collect_scene_motion

    pending = scoutdb.videos_missing_motion(database, MOTION_VERSION)
    total_scenes = sum(len(scenes) for _, _, scenes in pending)
    done_scenes = written = 0
    if on_progress:
        on_progress("motion", 0, total_scenes)

    for video_id, filepath, scenes in pending:
        if not Path(filepath).is_file():
            log(f"[diag] scene scout motion | source missing, skipped: {Path(filepath).name}")
            done_scenes += len(scenes)
            continue
        ranges = [(start, end) for _, start, end in scenes]
        measured = collect_scene_motion(
            filepath,
            ranges,
            on_progress=(lambda d, _t, base=done_scenes: on_progress("motion", base + d, total_scenes))
            if on_progress
            else None,
        )
        by_scene_index = {scenes[i][0]: value for i, value in measured.items()}
        written += scoutdb.insert_scene_motion(database, video_id, by_scene_index, MOTION_VERSION)
        done_scenes += len(scenes)
        log(f"[diag] scene scout motion | {Path(filepath).name}: {len(by_scene_index)} scenes")

    if on_progress:
        on_progress("done", total_scenes, total_scenes)
    return written


def model_version() -> str:
    return EMBEDDING_MODEL_VERSION


def thumbnail_needs_redo(thumbnail: Optional[bytes]) -> bool:
    """True for a missing thumbnail, or one smaller than this version stores.

    Databases from older tools (scene-scout-dev) stored 160x90 thumbnails, which
    look blurry stretched over a result tile; regenerating them is the upgrade.
    An unreadable blob counts as missing.
    """
    if thumbnail is None:
        return True
    from PIL import Image

    try:
        with Image.open(io.BytesIO(thumbnail)) as image:
            return image.width < THUMBNAIL_WIDTH
    except Exception:
        return True


def generate_missing_thumbnails(
    database: str | Path,
    *,
    on_progress: Optional[ProgressCallback] = None,
) -> int:
    """Fill in missing thumbnails and redo low-resolution ones.

    Returns how many thumbnails were written.
    """
    from ..infra.ipc import log

    db_path = scoutdb.resolve_db_path(database)
    if not db_path.is_file():
        raise FileNotFoundError(f"Database does not exist: {db_path}")

    with scoutdb.connect(db_path) as conn:
        # the blobs are small; reading them all is what lets old low-res ones be found
        todo_by_video: dict[int, list[tuple[int, int, int]]] = {}
        for scene_id, video_id, start_ms, end_ms, thumbnail in conn.execute(
            """
            SELECT id, video_id, start_time_ms, end_time_ms, thumbnail
            FROM scene_embeddings
            ORDER BY video_id, scene_index
            """
        ):
            if thumbnail_needs_redo(thumbnail):
                todo_by_video.setdefault(video_id, []).append((scene_id, start_ms, end_ms))

        total_missing = sum(len(rows) for rows in todo_by_video.values())
        if total_missing == 0:
            return 0

        filepaths = dict(conn.execute("SELECT id, filepath FROM processed_videos"))
        updated_count = 0
        # scenes dealt with so far, written or skipped, so the bar always reaches the end
        handled = 0
        log(
            f"[diag] scene scout generate-thumbnails | {total_missing} thumbnails to make "
            f"across {len(todo_by_video)} videos"
        )
        if on_progress:
            on_progress("thumbnails", 0, total_missing)

        for video_id, scene_rows in todo_by_video.items():
            filepath = filepaths.get(video_id)
            v_path = Path(filepath) if filepath else None
            if v_path is None or not v_path.is_file():
                log(f"[diag] scene scout generate-thumbnails | skipping missing video file: {filepath}")
                handled += len(scene_rows)
                if on_progress:
                    on_progress("thumbnails", handled, total_missing)
                continue

            log(f"[diag] scene scout generate-thumbnails | {v_path.name}: {len(scene_rows)} thumbnails")

            # decoding is the slow part, so progress follows the frames as they come
            def report_frames(done: int, _total: int, offset: int = handled) -> None:
                if on_progress:
                    on_progress("thumbnails", offset + done, total_missing)

            scenes = [(row[1], row[2]) for row in scene_rows]
            try:
                # thumbnails only: the default patch budget already keeps every frame thumbnail-wide
                frames = sample_frames(v_path, scenes, on_progress=report_frames, max_patches=256)
            except Exception as e:
                log(f"[diag] scene scout generate-thumbnails | decode error on {v_path.name}: {e}")
                handled += len(scene_rows)
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
            handled += len(scene_rows)
            if on_progress:
                on_progress("thumbnails", handled, total_missing)

        return updated_count
