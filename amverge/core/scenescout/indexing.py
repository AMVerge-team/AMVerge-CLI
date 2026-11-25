"""Turning a video into rows in a Scene Scout database (schema v4).

Pipeline: detect scene boundaries, sample a representative frame per scene
without decoding the rest, embed the frames, write scenes + contiguous float32
``video_embeddings`` + 320w WebP ``scene_thumbnails``.
"""

from __future__ import annotations

import bisect
import io
import queue
import struct
import threading
from pathlib import Path
from typing import Callable, Optional

from . import db as scoutdb
from .fingerprint import compute_content_fingerprint
from .types import EMBEDDING_MODEL_VERSION

ProgressCallback = Callable[[str, int, int], None]

# 320w WebP keeps multi-season databases compact while looking sharp across both apps
THUMBNAIL_WIDTH = 320
THUMBNAIL_QUALITY = 75

SAMPLING_STRATEGY_V1 = "first_or_midpoint_v1"

KEYFRAME_SNAP_FRAMES = 2
BLANK_CHECK_WIDTH = 320
BLANK_DETAIL_THRESHOLD = 1.5


# the app's settings values, passed straight through
DETECTOR_AI = "transnetv2_gpu"
DETECTOR_KEYFRAME = "keyframe_detection"


def _detect_scenes_with_label(
    video: str | Path,
    *,
    method: str = DETECTOR_KEYFRAME,
    accurate: bool = False,
) -> tuple[list[tuple[int, int]], str]:
    """Return ``(scenes_ms, canonical_detector_name)`` ('transnetv2' or 'keyframe')."""
    use_ai = accurate or method in (DETECTOR_AI, "transnetv2")
    if use_ai:
        from ..detection.ai_scene_detection import TRANSNET_AVAILABLE, decode_and_detect_scenes

        if TRANSNET_AVAILABLE:
            scenes_secs, _ = decode_and_detect_scenes(video)
            scenes = [(int(s * 1000), int(e * 1000)) for s, e in scenes_secs]
            if scenes:
                return scenes, "transnetv2"

    from ..detection.keyframe import detect_scenes_by_keyframe

    keyframe_scenes = detect_scenes_by_keyframe(video)
    return [(int(s * 1000), int(e * 1000)) for s, e in keyframe_scenes], "keyframe"


def detect_scenes(
    video: str | Path,
    *,
    method: str = DETECTOR_KEYFRAME,
    accurate: bool = False,
) -> list[tuple[int, int]]:
    """Scene boundaries as ``(start_ms, end_ms)`` pairs."""
    scenes, _ = _detect_scenes_with_label(video, method=method, accurate=accurate)
    return scenes


def probe_video_metadata(video: str | Path) -> tuple[Optional[int], Optional[float], Optional[int], Optional[int]]:
    """Return ``(duration_ms, fps, width, height)`` from container headers without decoding."""
    import av

    try:
        with av.open(str(video), options={"err_detect": "ignore_err"}) as container:
            if not container.streams.video:
                return None, None, None, None
            stream = container.streams.video[0]
            fps = float(stream.average_rate) if stream.average_rate and stream.average_rate > 0 else None
            width = int(stream.codec_context.width or stream.width or 0) or None
            height = int(stream.codec_context.height or stream.height or 0) or None
            duration_ms: Optional[int] = None
            if stream.duration is not None and stream.time_base is not None:
                duration_ms = int(float(stream.duration * stream.time_base) * 1000)
            elif container.duration is not None:
                duration_ms = int((container.duration / av.time_base) * 1000)
            return duration_ms, round(fps, 4) if fps else None, width, height
    except Exception:
        return None, None, None, None


def _sample_size(width: int, height: int, max_patches: int) -> tuple[int, int]:
    """The size a sampled frame is shrunk to: never bigger than the source."""
    target_area = max_patches * 16 * 16 * 4
    scale = min(1.0, max((target_area / (width * height)) ** 0.5, THUMBNAIL_WIDTH / width))
    return max(1, round(width * scale)), max(1, round(height * scale))


def sample_frames(
    video: str | Path,
    scenes: list[tuple[int, int]],
    on_progress: Optional[Callable[[int, int], None]] = None,
    max_patches: Optional[int] = None,
) -> list:
    """The middle frame of each scene, decoding the video in full."""
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

    midpoints_ms = [(start_ms + end_ms) // 2 for start_ms, end_ms in scenes]

    from .decode import close_video, open_video

    container = open_video(video)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    def to_rgb(frame):
        if max_patches is None:
            return frame.to_ndarray(format="rgb24")
        width, height = _sample_size(frame.width, frame.height, max_patches)
        return frame.reformat(width=width, height=height, format="rgb24", interpolation="AREA").to_ndarray()

    out: list = []
    next_midpoint = 0
    last_raw = None

    try:
        for frame in container.decode(stream):
            try:
                current_ms = int(frame.time * 1000)
            except (TypeError, ValueError):
                continue
            last_raw = frame

            if current_ms < midpoints_ms[next_midpoint]:
                continue
            try:
                nd = to_rgb(frame)
            except (av.FFmpegError, ValueError):
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
    """True for a frame with little to search for: a flash, black, or an empty background."""
    import numpy as np

    stride = max(1, frame.shape[1] // BLANK_CHECK_WIDTH)
    gray = frame[::stride, ::stride].mean(axis=2)
    detail = (np.abs(np.diff(gray, axis=1)).mean() + np.abs(np.diff(gray, axis=0)).mean()) / 2
    return bool(detail < BLANK_DETAIL_THRESHOLD)


def _packet_times(video: str | Path) -> tuple[list[int], list[int], float]:
    """Keyframe times and all frame times in ms, sorted, plus the frame rate."""
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
    """The exact frame time to sample for each scene: its first frame, or the keyframe it sits on."""
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
    """One representative frame per scene, without decoding the whole video."""
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
    """320w WebP bytes for the ``scene_thumbnails.thumbnail`` BLOB column."""
    from PIL import Image

    image = Image.fromarray(frame)
    if image.width > THUMBNAIL_WIDTH:
        height = max(1, round(image.height * THUMBNAIL_WIDTH / image.width))
        image = image.resize((THUMBNAIL_WIDTH, height), Image.LANCZOS)

    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="WEBP", quality=THUMBNAIL_QUALITY, method=0)
    return buffer.getvalue()


def _fast_image_width(blob: bytes) -> Optional[int]:
    """Read WebP or JPEG width directly from header bytes without full image decode."""
    n = len(blob)
    # WebP: RIFF <size> WEBP <chunk>
    if n >= 30 and blob[:4] == b"RIFF" and blob[8:12] == b"WEBP":
        chunk = blob[12:16]
        if chunk == b"VP8 " and n >= 30 and blob[23:26] == b"\x9d\x01\x2a":
            return struct.unpack_from("<H", blob, 26)[0] & 0x3FFF
        if chunk == b"VP8L" and n >= 25 and blob[20] == 0x2F:
            bits = struct.unpack_from("<I", blob, 21)[0]
            return (bits & 0x3FFF) + 1
        if chunk == b"VP8X" and n >= 30:
            return (blob[24] | (blob[25] << 8) | (blob[26] << 16)) + 1
    # JPEG: FF D8 ... SOF marker
    if n >= 4 and blob[:2] == b"\xff\xd8":
        i = 2
        while i + 9 < n:
            if blob[i] != 0xFF:
                break
            while i < n and blob[i] == 0xFF:
                i += 1
            if i >= n:
                break
            marker = blob[i]
            i += 1
            if marker in (0xD8, 0xD9) or (0xD0 <= marker <= 0xD7):
                continue
            if i + 2 > n:
                break
            seg_len = struct.unpack_from(">H", blob, i)[0]
            if seg_len < 2 or i + seg_len > n:
                break
            if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                if i + 7 <= n:
                    return struct.unpack_from(">H", blob, i + 5)[0]
                break
            i += seg_len
    return None


def precut_scenes(raw) -> list[tuple[int, int]]:
    """Scene ranges handed in by the caller instead of detected, as ``(start_ms, end_ms)``."""
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
    """Index one video into a database. Returns the number of scenes written."""
    from ..infra.ipc import log
    from .embedding import is_model_loaded

    video_path = Path(video).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"No such video: {video_path}")

    def report(stage: str, done: int, total: int) -> None:
        if on_progress:
            on_progress(stage, done, total)

    if scenes:
        effective_detector = "precut"
        log(f"[diag] scene scout {video_path.name} | using {len(scenes)} pre-cut scenes, detection skipped")
    else:
        report("detecting", 0, 1)
        log(f"[diag] scene scout {video_path.name} | detecting scenes using {method}...")
        scenes, effective_detector = _detect_scenes_with_label(video_path, method=method, accurate=accurate)
        if not scenes:
            log(f"[diag] scene scout {video_path.name} | no scenes detected")
            return 0
        log(f"[diag] scene scout {video_path.name} | detected {len(scenes)} scenes ({effective_detector})")

    file_size, content_fp = compute_content_fingerprint(video_path)
    duration_ms, fps, width, height = probe_video_metadata(video_path)

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

    sampling_cfg = {
        "frames_per_scene": 1,
        "pooling": "none",
        "max_patches": int(max_patches),
    }
    video_id = scoutdb.upsert_video(
        database,
        str(video_path),
        video_path.stat().st_mtime,
        status="indexing",
        model_id=EMBEDDING_MODEL_VERSION,
        detector=effective_detector,
        sampling_strategy=SAMPLING_STRATEGY_V1,
        sampling_config=sampling_cfg,
        file_size=file_size,
        content_fingerprint=content_fp,
        duration_ms=duration_ms,
        fps=fps,
        width=width,
        height=height,
    )

    thumbnails = [encode_thumbnail(f) for f in frames] if generate_thumbnails else None
    scoutdb.insert_video_scenes(database, video_id, scenes, vectors, thumbnails)
    scoutdb.mark_video_complete(database, video_id)
    report("done", len(scenes), len(scenes))
    log(f"[diag] scene scout {video_path.name} | indexed {len(scenes)} scenes successfully")

    return len(scenes)


_END = object()


class _BackgroundEmbedder:
    """Embeds sampled frames on a worker thread, a batch at a time, while the decode runs."""

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
        reported = 0
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
    """Measures motion for every indexed video that lacks it. Returns scenes written."""
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


def thumbnail_needs_redo(thumbnail: Optional[bytes], *, upgrade_low_res: bool = True) -> bool:
    """True for a missing/corrupt thumbnail, or (when ``upgrade_low_res``) narrower than :data:`THUMBNAIL_WIDTH`."""
    if not thumbnail:
        return True

    fast_w = _fast_image_width(thumbnail)
    if fast_w is not None:
        return fast_w < THUMBNAIL_WIDTH if upgrade_low_res else False

    from PIL import Image

    try:
        with Image.open(io.BytesIO(thumbnail)) as image:
            w = image.width
            return w < THUMBNAIL_WIDTH if upgrade_low_res else False
    except Exception:
        return True


def generate_missing_thumbnails(
    database: str | Path,
    *,
    upgrade_low_res: bool = True,
    on_progress: Optional[ProgressCallback] = None,
) -> int:
    """Fill in missing thumbnails (and optionally redo low-resolution ones) in ``scene_thumbnails``."""
    from ..infra.ipc import log

    db_path = scoutdb.resolve_db_path(database)
    if not db_path.is_file():
        raise FileNotFoundError(f"Database does not exist: {db_path}")

    scoutdb.ensure_v4_ready(db_path)

    updated_count = 0
    with scoutdb.connect(db_path) as conn:
        todo_by_video: dict[int, list[tuple[int, int, int]]] = {}
        # Fetch only the 64-byte image header prefix first so we never load tens of MB of BLOBs just to check width
        for scene_id, video_id, start_ms, end_ms, header_prefix in conn.execute(
            """
            SELECT s.id, s.video_id, s.start_time_ms, s.end_time_ms, SUBSTR(t.thumbnail, 1, 64)
            FROM scenes s
            LEFT JOIN scene_thumbnails t ON t.scene_id = s.id
            ORDER BY s.video_id, s.scene_index
            """
        ):
            if header_prefix is None:
                needs = True
            else:
                fast_w = _fast_image_width(bytes(header_prefix))
                if fast_w is not None:
                    needs = fast_w < THUMBNAIL_WIDTH if upgrade_low_res else False
                else:
                    full_row = conn.execute(
                        "SELECT thumbnail FROM scene_thumbnails WHERE scene_id = ?", (scene_id,)
                    ).fetchone()
                    needs = thumbnail_needs_redo(full_row[0] if full_row else None, upgrade_low_res=upgrade_low_res)
            if needs:
                todo_by_video.setdefault(video_id, []).append((scene_id, start_ms, end_ms))

        total_missing = sum(len(rows) for rows in todo_by_video.values())
        if total_missing == 0:
            return 0

        filepaths = dict(conn.execute("SELECT id, filepath FROM processed_videos"))
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

            def report_frames(done: int, _total: int, offset: int = handled) -> None:
                if on_progress:
                    on_progress("thumbnails", offset + done, total_missing)

            scenes = [(row[1], row[2]) for row in scene_rows]
            try:
                frames = sample_frames(v_path, scenes, on_progress=report_frames, max_patches=256)
            except Exception as e:
                log(f"[diag] scene scout generate-thumbnails | decode error on {v_path.name}: {e}")
                handled += len(scene_rows)
                continue

            batch_updates = []
            for (scene_id, _, _), frame in zip(scene_rows, frames):
                thumb_bytes = encode_thumbnail(frame)
                batch_updates.append((scene_id, thumb_bytes))

            conn.executemany(
                "INSERT OR REPLACE INTO scene_thumbnails (scene_id, thumbnail) VALUES (?, ?)",
                batch_updates,
            )
            conn.commit()

            updated_count += len(batch_updates)
            handled += len(scene_rows)
            if on_progress:
                on_progress("thumbnails", handled, total_missing)

        if updated_count > 0:
            scoutdb._touch_updated_metadata(conn)

    if updated_count > 0:
        scoutdb.run_incremental_vacuum(db_path)
    return updated_count
