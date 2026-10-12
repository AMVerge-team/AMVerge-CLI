"""Motion tags for Scene Scout: how each scene starts and ends moving.

SigLIP sees one middle frame per scene, so it knows what a scene shows but not
which way it moves. This module measures the first and last ~0.4s of every
scene with classical optical flow (OpenCV LK, no model) and reduces it to a
few raw numbers per boundary:

* camera: the whole-image transform (pan x/y, zoom) fitted with RANSAC
* subject: on-screen velocity of the points that move independently of it

A dense-flow variant (DIS at 320px, with cut trimming) was tried and read
more tags wrong in real use than this one, so sparse tracking stays.

Raw numbers are what the database stores. Tags (``right``, ``pan-left``,
``zoom-in``, ``fast``, ``unclear``...) are derived from them on every read by
:func:`tags_for`, so thresholds can be tuned without re-indexing anything.

Velocities are in frame widths per second, so they do not depend on the
resolution the frames were analysed at. +x is right, +y is down.
"""

from __future__ import annotations

import math
import re
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Optional

import numpy as np

# bumped when the measurement itself changes; rows from an older version are
# re-measured by `scout motion` instead of being compared against new ones
# back at 2: the v2 measurement is restored, so v3 rows read as stale and are re-measured
MOTION_VERSION = 2

# mean grey-level difference below which a frame is a held drawing, not a new one
HELD_FRAME_DIFF = 0.6

# how much of each scene end is measured, and at what size
BOUNDARY_MS = 420
MAX_BOUNDARY_FRAMES = 10
ANALYSIS_WIDTH = 480

# tag thresholds, all in frame widths per second (zoom: log-scale per second)
PAN_MIN = 0.05
ZOOM_MIN = 0.10
STILL_SPEED = 0.06
FAST_SPEED = 0.60
MEDIUM_SPEED = 0.22
MIN_CAMERA_CONFIDENCE = 0.35
MIN_SUBJECT_SUPPORT = 10
MIN_SUBJECT_CONSISTENCY = 0.55

DIRECTIONS = ("right", "down-right", "down", "down-left", "left", "up-left", "up", "up-right")
SUBJECT_TAGS = DIRECTIONS + ("still", "unclear")
CAMERA_TAGS = tuple(f"pan-{d}" for d in DIRECTIONS) + ("zoom-in", "zoom-out", "static", "unclear")
SPEED_TAGS = ("slow", "medium", "fast")

Raw = dict[str, Any]


def _cv2():
    import cv2

    return cv2


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------


def measure_boundary(frames: list[np.ndarray], times: list[float]) -> Optional[Raw]:
    """Camera and subject motion over a few consecutive greyscale frames.

    ``times`` are the frames' timestamps in seconds. Returns None when there is
    nothing to measure (too few frames, or a texture-less image).
    """
    cv2 = _cv2()
    if len(frames) != len(times):
        return None
    frames, times = _drop_held_frames(frames, times)
    if len(frames) < 3:
        return None

    height, width = frames[0].shape[:2]
    pts = cv2.goodFeaturesToTrack(frames[0], maxCorners=400, qualityLevel=0.01, minDistance=6)
    if pts is None or len(pts) < 12:
        return None

    # a residual this large (pixels, per frame) counts as moving independently of the camera
    move_px = max(1.0, 0.004 * width)

    cam_v, zoom_v, inlier_ratios = [], [], []
    subj_v, subj_rel_v, supports = [], [], []
    pairs = 0

    for (a, ta), (b, tb) in zip(zip(frames, times), zip(frames[1:], times[1:])):
        dt = tb - ta
        if dt <= 0:
            continue
        p1, st, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None, winSize=(21, 21), maxLevel=3)  # type: ignore[call-overload]
        p0r, st2, _ = cv2.calcOpticalFlowPyrLK(b, a, p1, None, winSize=(21, 21), maxLevel=3)  # type: ignore[call-overload]
        ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(pts - p0r, axis=2).ravel() < 1.0)
        if ok.sum() < 12:
            break
        p0g, p1g = pts[ok], p1[ok]
        matrix, inliers = cv2.estimateAffinePartial2D(p0g, p1g, method=cv2.RANSAC, ransacReprojThreshold=2.0)
        if matrix is None:
            break
        pairs += 1
        # pan is how far the frame centre moves: the matrix's own translation is about
        # the top-left corner, so a centred zoom would otherwise read as a pan
        cx, cy = width / 2, height / 2
        moved_x = matrix[0, 0] * cx + matrix[0, 1] * cy + matrix[0, 2] - cx
        moved_y = matrix[1, 0] * cx + matrix[1, 1] * cy + matrix[1, 2] - cy
        cam_v.append((moved_x / width / dt, moved_y / width / dt))
        zoom_v.append(math.log(max(1e-6, math.hypot(matrix[0, 0], matrix[1, 0]))) / dt)
        inlier_ratios.append(float(inliers.mean()) if inliers is not None else 0.0)

        flat0, flat1 = p0g.reshape(-1, 2), p1g.reshape(-1, 2)
        residual = flat1 - cv2.transform(p0g, matrix).reshape(-1, 2)
        moving = np.linalg.norm(residual, axis=1) > move_px
        supports.append(int(moving.sum()))
        if moving.sum() >= MIN_SUBJECT_SUPPORT:
            # screen velocity of the independently moving points: what the viewer sees move
            subj_v.append(np.median(flat1[moving] - flat0[moving], axis=0) / width / dt)
            subj_rel_v.append(np.median(residual[moving], axis=0) / width / dt)

        pts = p1g.reshape(-1, 1, 2)

    if pairs == 0:
        return None

    raw: Raw = {
        "v": MOTION_VERSION,
        "pairs": pairs,
        "cam": [_r(np.mean([c[0] for c in cam_v])), _r(np.mean([c[1] for c in cam_v])), _r(np.mean(zoom_v))],
        "camConf": _r(float(np.mean(inlier_ratios))),
        "subjPairs": len(subj_v),
        "subjSupport": int(np.median(supports)) if supports else 0,
    }
    if subj_v:
        vectors = np.array(subj_v)
        raw["subj"] = [_r(v) for v in vectors.mean(axis=0)]
        raw["subjRel"] = [_r(v) for v in np.array(subj_rel_v).mean(axis=0)]
        raw["subjConsistency"] = _r(_consistency(vectors))
    return raw

_PLANAR_YUV = re.compile(r"^yuv(j?)4[024]{2}p(?:(\d+)(le|be))?$")


def frame_to_grey(frame, width: int, height: int) -> np.ndarray:
    """A decoded PyAV frame as a small 8-bit greyscale image.

    Planar YUV already stores brightness as its own plane, so this reads that
    plane and shrinks it with OpenCV instead of running a full colour
    conversion: about twice as fast, and boundary frames are converted on the
    decode thread, where every millisecond adds to indexing time. Limited-range
    video (16-235) is stretched to 0-255 so thresholds mean the same either way.
    Anything else falls back to PyAV's converter.
    """
    cv2 = _cv2()
    match = _PLANAR_YUV.match(frame.format.name)
    if not match or match.group(3) == "be":
        return frame.reformat(width=width, height=height, format="gray").to_ndarray()

    full_range, bits = match.group(1) == "j", int(match.group(2) or 8)
    plane = frame.planes[0]
    if bits == 8:
        luma = np.frombuffer(plane, dtype=np.uint8).reshape(frame.height, plane.line_size)[:, : frame.width]
    else:
        stride = plane.line_size // 2
        wide = np.frombuffer(plane, dtype=np.uint16).reshape(frame.height, stride)[:, : frame.width]
        luma = (wide >> (bits - 8)).astype(np.uint8)
    small = cv2.resize(luma, (width, height), interpolation=cv2.INTER_AREA)
    if full_range:
        return small
    return cv2.convertScaleAbs(small, alpha=255.0 / 219.0, beta=-16.0 * 255.0 / 219.0)


def _analysis_size(frame) -> tuple[int, int]:
    return ANALYSIS_WIDTH, max(2, round(frame.height * ANALYSIS_WIDTH / frame.width / 2) * 2)


def _drop_held_frames(frames: list[np.ndarray], times: list[float]) -> tuple[list[np.ndarray], list[float]]:
    """Keep one frame per drawing.

    Anime is animated on twos and threes: each drawing is held for 2-3 frames.
    Measured frame to frame, half the pairs show no motion at all and a moving
    character reads as still. Measured drawing to drawing, across the real time
    between them, the velocity is right. A truly static shot keeps its first and
    last frame so it still has a measurable (zero) motion.
    """
    kept_frames, kept_times = [frames[0]], [times[0]]
    for frame, t in zip(frames[1:], times[1:]):
        if float(np.mean(np.abs(frame.astype(np.int16) - kept_frames[-1]))) >= HELD_FRAME_DIFF:
            kept_frames.append(frame)
            kept_times.append(t)
    if len(kept_frames) < 3 and len(frames) >= 3:
        # held or still throughout: measure across the whole window instead
        middle = len(frames) // 2
        return [frames[0], frames[middle], frames[-1]], [times[0], times[middle], times[-1]]
    return kept_frames, kept_times


def _consistency(vectors: np.ndarray) -> float:
    """1 when every per-frame velocity points the same way, near 0 when they cancel out."""
    norms = np.linalg.norm(vectors, axis=1)
    keep = norms > 1e-6
    if not keep.any():
        return 0.0
    units = vectors[keep] / norms[keep, None]
    return float(np.linalg.norm(units.mean(axis=0)))


def _r(value: float) -> float:
    return round(float(value), 4)


# ---------------------------------------------------------------------------
# tags
# ---------------------------------------------------------------------------


def direction8(vx: float, vy: float) -> str:
    """Screen direction of a velocity, in eight 45 degree sectors (+y is down)."""
    angle = math.degrees(math.atan2(vy, vx)) % 360
    return DIRECTIONS[int(((angle + 22.5) % 360) // 45)]


def _opposite(direction: str) -> str:
    return DIRECTIONS[(DIRECTIONS.index(direction) + 4) % 8]


def _speed_tag(speed: float) -> str:
    if speed >= FAST_SPEED:
        return "fast"
    if speed >= MEDIUM_SPEED:
        return "medium"
    return "slow"


def camera_tag(raw: Optional[Raw]) -> str:
    if not raw or raw.get("pairs", 0) < 2 or raw.get("camConf", 0.0) < MIN_CAMERA_CONFIDENCE:
        return "unclear"
    vx, vy, zoom = raw["cam"]
    pan = math.hypot(vx, vy)
    # zoom wins when it explains more of the image change than the translation does
    if abs(zoom) >= ZOOM_MIN and abs(zoom) * 0.5 >= pan:
        return "zoom-in" if zoom > 0 else "zoom-out"
    if pan >= PAN_MIN:
        # image content moving right means the camera is panning left
        return f"pan-{_opposite(direction8(vx, vy))}"
    return "static"


def subject_tag(raw: Optional[Raw]) -> tuple[str, Optional[str]]:
    """(direction | still | unclear, speed bucket or None)."""
    if not raw or raw.get("pairs", 0) < 2:
        return "unclear", None
    pairs, moving_pairs = raw["pairs"], raw.get("subjPairs", 0)
    if moving_pairs * 2 < pairs:
        # nothing moves on its own for most of the window; only trust that over a solid camera fit
        return ("still", None) if camera_tag(raw) != "unclear" else ("unclear", None)
    if raw.get("subjConsistency", 0.0) < MIN_SUBJECT_CONSISTENCY or raw.get("subjSupport", 0) < MIN_SUBJECT_SUPPORT:
        return "unclear", None
    vx, vy = raw["subj"]
    speed = math.hypot(vx, vy)
    if speed < STILL_SPEED:
        return "still", None
    return direction8(vx, vy), _speed_tag(speed)


def tags_for(raw: Optional[Raw]) -> dict[str, Any]:
    subject, speed = subject_tag(raw)
    return {"subject": subject, "camera": camera_tag(raw), "speed": speed}


def scene_tags(start: Optional[Raw], end: Optional[Raw]) -> dict[str, Any]:
    return {"start": tags_for(start), "end": tags_for(end)}


# ---------------------------------------------------------------------------
# collecting boundary frames during a decode that already happens
# ---------------------------------------------------------------------------


class BoundaryCollector:
    """Keeps each scene's first and last ~0.4s while a video decodes.

    Fed every decoded frame by the sampler. A scene's two boundaries are
    measured on worker threads as soon as the decode passes its end, and their
    frames are dropped, so memory stays at a few scenes' worth however long the
    video is. OpenCV releases the GIL, so the threads run in parallel.
    """

    def __init__(self, scenes: list[tuple[int, int]], workers: int = 8, max_pending: int = 48):
        self.scenes = scenes
        self.results: dict[int, tuple[Optional[Raw], Optional[Raw]]] = {}
        self._starts: dict[int, list] = {}
        self._ends: dict[int, list] = {}
        self._index = 0
        self._size: Optional[tuple[int, int]] = None
        self._pool = ThreadPoolExecutor(max_workers=workers)
        self._pending: list[tuple[int, Future, Future]] = []
        self._max_pending = max_pending

    def wants(self, t_ms: int) -> bool:
        """Whether a frame at ``t_ms`` belongs to a boundary; lets the sampler skip conversion."""
        self._advance(t_ms)
        if self._index >= len(self.scenes):
            return False
        start, end = self.scenes[self._index]
        in_start = t_ms < start + BOUNDARY_MS and len(self._starts.get(self._index, ())) < MAX_BOUNDARY_FRAMES
        return in_start or t_ms >= end - BOUNDARY_MS

    def add(self, t_ms: int, frame) -> None:
        """Store a decoded PyAV frame (only called when :meth:`wants` said yes)."""
        if self._size is None:
            self._size = _analysis_size(frame)
        grey = frame_to_grey(frame, *self._size)
        start, end = self.scenes[self._index]
        if t_ms < start + BOUNDARY_MS:
            self._starts.setdefault(self._index, []).append((grey, t_ms / 1000.0))
        if t_ms >= end - BOUNDARY_MS:
            tail = self._ends.setdefault(self._index, [])
            tail.append((grey, t_ms / 1000.0))
            if len(tail) > MAX_BOUNDARY_FRAMES:
                tail.pop(0)

    def _advance(self, t_ms: int) -> None:
        while self._index < len(self.scenes) and t_ms >= self.scenes[self._index][1]:
            self._submit(self._index)
            self._index += 1

    def _submit(self, index: int) -> None:
        starts, ends = self._starts.pop(index, []), self._ends.pop(index, [])
        submit = lambda items: self._pool.submit(measure_boundary, [f for f, _ in items], [t for _, t in items])
        self._pending.append((index, submit(starts), submit(ends)))
        # backpressure: the decode never runs far ahead of the measurement
        while len(self._pending) > self._max_pending:
            self._collect(self._pending.pop(0))

    def _collect(self, entry: tuple[int, Future, Future]) -> None:
        index, start, end = entry
        self.results[index] = (_safe(start), _safe(end))

    def finish(self, on_progress: Optional[Callable[[int, int], None]] = None) -> dict[int, tuple[Optional[Raw], Optional[Raw]]]:
        """Measures whatever the decode left pending and returns ``{scene_index: (start, end)}``."""
        while self._index < len(self.scenes):
            self._submit(self._index)
            self._index += 1
        total = len(self.scenes)
        for entry in self._pending:
            self._collect(entry)
            if on_progress:
                on_progress(len(self.results), total)
        self._pending.clear()
        self._pool.shutdown(wait=True)
        if on_progress:
            on_progress(total, total)
        return self.results


def _safe(future: Future) -> Optional[Raw]:
    try:
        return future.result()
    except Exception:  # noqa: BLE001 - one bad boundary leaves that scene unclear, never fails the index
        return None


def collect_scene_motion(
    video: str,
    scenes: list[tuple[int, int]],
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> dict[int, tuple[Optional[Raw], Optional[Raw]]]:
    """Decodes ``video`` once and measures every scene's boundaries (used by backfill)."""
    import av

    collector = BoundaryCollector(scenes)
    container = av.open(str(video), options={"err_detect": "ignore_err"})
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    step = max(1, len(scenes) // 100)
    last = -1
    try:
        for frame in container.decode(stream):
            try:
                t_ms = int(frame.time * 1000)
            except (TypeError, ValueError):
                continue
            if collector.wants(t_ms):
                collector.add(t_ms, frame)
            if on_progress and collector._index - last >= step:
                last = collector._index
                on_progress(min(collector._index, len(scenes)), len(scenes))
            if collector._index >= len(scenes):
                break
    finally:
        container.close()
    return collector.finish()


# ---------------------------------------------------------------------------
# a single query clip
# ---------------------------------------------------------------------------


def measure_clip(path: str, start_ms: Optional[int] = None, end_ms: Optional[int] = None) -> dict[str, Any]:
    """Start/end motion of one clip (or a range inside a longer video).

    Also returns the range's last frame as RGB, which a flow search embeds with
    SigLIP when there is no text query. Seeks, so the cost is the two short
    windows, not the whole file.
    """
    import av

    container = av.open(str(path), options={"err_detect": "ignore_err"})
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        duration_ms = int(float(stream.duration * stream.time_base) * 1000) if stream.duration else None
        if duration_ms is None and container.duration:
            duration_ms = int(container.duration / 1000)
        lo = start_ms or 0
        hi = end_ms if end_ms is not None else duration_ms
        if hi is None:
            raise ValueError("Could not read the clip's duration; pass --end-ms")

        start_frames = _read_window(container, stream, lo, min(hi, lo + BOUNDARY_MS))
        end_frames = _read_window(container, stream, max(lo, hi - BOUNDARY_MS), hi, keep_rgb_last=True)
    finally:
        container.close()

    last_rgb = end_frames.pop("last_rgb", None)
    start_raw = measure_boundary([f for f, _ in start_frames["grey"]], [t for _, t in start_frames["grey"]])
    end_raw = measure_boundary([f for f, _ in end_frames["grey"]], [t for _, t in end_frames["grey"]])
    return {"start": start_raw, "end": end_raw, "lastFrame": last_rgb}


def _read_window(container, stream, lo_ms: int, hi_ms: int, keep_rgb_last: bool = False) -> dict[str, Any]:
    container.seek(max(0, lo_ms - 1000) * 1000, any_frame=False)
    size = None
    grey: list = []
    last = None
    for frame in container.decode(stream):
        try:
            t_ms = int(frame.time * 1000)
        except (TypeError, ValueError):
            continue
        if t_ms < lo_ms:
            continue
        if t_ms >= hi_ms and grey:
            break
        if size is None:
            size = _analysis_size(frame)
        grey.append((frame_to_grey(frame, *size), t_ms / 1000.0))
        last = frame
        if len(grey) >= MAX_BOUNDARY_FRAMES + 2 and not keep_rgb_last:
            break
    grey = grey[-MAX_BOUNDARY_FRAMES:] if keep_rgb_last else grey[:MAX_BOUNDARY_FRAMES]
    out: dict[str, Any] = {"grey": grey}
    if keep_rgb_last and last is not None:
        out["last_rgb"] = last.to_ndarray(format="rgb24")
    return out
