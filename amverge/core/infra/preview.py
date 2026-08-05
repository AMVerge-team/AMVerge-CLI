from __future__ import annotations

"""Preview-frame emitter for IPC mode.

Writes throttled JPEG snapshots of in-progress frames and emits
``PREVIEW_FRAME|<tag>|<path>|<seq>`` events on stderr for a consuming app
to display. Ping-pongs between two files per tag so a half-written JPEG is
never read by the other side. No-op when OpenCV is unavailable.
"""

import os
import tempfile
import threading

from .ipc import emit_event


class PreviewEmitter:
    def __init__(
        self,
        tag: str,
        out_dir: str | None = None,
        min_pct_delta: float = 2.0,
        max_width: int = 640,
    ) -> None:
        self.tag = tag
        self.dir = out_dir or os.path.join(tempfile.gettempdir(), "amverge_preview")
        os.makedirs(self.dir, exist_ok=True)
        self.min_delta = min_pct_delta
        self.max_width = max_width
        self._last_pct = -1e9
        self._seq = 0
        self._slot = 0
        self._lock = threading.Lock()

    def maybe_emit(self, frame_bgr, pct: float) -> None:
        if frame_bgr is None:
            return
        if pct - self._last_pct < self.min_delta:
            return
        self._last_pct = pct
        self.emit(frame_bgr)

    def emit(self, frame_bgr) -> None:
        try:
            import cv2
        except ImportError:
            return
        with self._lock:
            self._seq += 1
            self._slot ^= 1
            path = os.path.join(self.dir, f"{self.tag}_{self._slot}.jpg")
            try:
                frame = frame_bgr
                h, w = frame.shape[:2]
                if w > self.max_width:
                    scale = self.max_width / float(w)
                    frame = cv2.resize(
                        frame, (self.max_width, max(1, int(h * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                ok = cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
            except Exception:
                ok = False
            if ok:
                emit_event(f"PREVIEW_FRAME|{self.tag}|{path}|{self._seq}")


def ipc_callbacks(tag: str):
    """Return ``(progress_cb, preview_cb)`` that stream IPC events for ``tag``.

    ``progress_cb(pct, msg)`` emits ``PROGRESS|`` and ``preview_cb(frame, pct)``
    emits throttled ``PREVIEW_FRAME|`` events. Use in ``--ipc`` command paths.
    """
    from .ipc import emit_progress

    emitter = PreviewEmitter(tag)

    def progress_cb(pct: int, msg: str) -> None:
        emit_progress(pct, msg)

    def preview_cb(frame, pct: int) -> None:
        emitter.maybe_emit(frame, pct)

    return progress_cb, preview_cb
