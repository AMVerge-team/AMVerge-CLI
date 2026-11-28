from __future__ import annotations

"""TransNetV2 scene detection.

Decodes video frames and runs TransNetV2 CNN inference to detect scene
boundaries. Supports two decode paths: Nelux (Windows native, optional) and
FFmpeg pipe (cross-platform).

Usage:
    >>> from amverge.core.detection.ai_scene_detection import decode_and_detect_scenes
    >>> scenes_secs, scenes_frames = decode_and_detect_scenes("episode.mp4")
    >>> print(f"Detected {len(scenes_secs)} scenes")
"""

import contextlib
import importlib.util
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np

from ..infra.binaries import get_ffmpeg
from ..infra.ipc import emit_progress, log
from .nelux_runtime import _get_nelux_video_reader
from ..video.probe_utils import probe_video_codec, probe_video_fps, probe_video_duration, probe_video_total_frames

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
from ..video.scene_utils import transnet_scenes_to_seconds
from ..transnet.transnet_constants import (
    FRAME_BYTES,
    FRAME_CHANNELS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    WINDOW_SIZE,
    STRIDE,
)

TRANSNET_AVAILABLE = (
    importlib.util.find_spec("transnetv2_pytorch") is not None
    and importlib.util.find_spec("torch") is not None
)

DEFAULT_THRESHOLD = 0.5

NVDEC_CODECS = frozenset({"h264", "hevc", "av1", "vp9"})

NELUX_STALL_SECONDS = 10.0

_TRANSNET_LOCK = threading.RLock()
_TRANSNET_MODEL = None
_TRANSNET_DEVICE: str | None = None


def get_transnet_model(device: str):
    """Return a cached ``TransNetV2`` instance on ``device`` (loaded once per process/device)."""
    global _TRANSNET_MODEL, _TRANSNET_DEVICE
    from transnetv2_pytorch import TransNetV2

    with _TRANSNET_LOCK:
        if _TRANSNET_MODEL is not None and _TRANSNET_DEVICE == device:
            return _TRANSNET_MODEL
        with contextlib.redirect_stdout(sys.stderr):
            model = TransNetV2(device=device)
        model.eval()
        _TRANSNET_MODEL = model
        _TRANSNET_DEVICE = device
        return model


def unload_transnet_model() -> bool:
    """Release the cached ``TransNetV2`` model from memory/VRAM."""
    global _TRANSNET_MODEL, _TRANSNET_DEVICE
    with _TRANSNET_LOCK:
        if _TRANSNET_MODEL is None:
            return False
        import gc
        import torch

        _TRANSNET_MODEL = None
        _TRANSNET_DEVICE = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return True


def _safe_total(total_frames: int) -> int:
    return max(1, int(total_frames) if total_frames else 1)


def _emit_loop_progress(
    processed: int,
    total: int,
    base: int,
    span: int,
    prefix: str,
    last: int,
) -> int:
    fraction = min(1.0, max(0.0, processed / _safe_total(total)))
    current = int(base + fraction * span)
    # long episodes move a whole percent only every few thousand frames, so the
    # frame count is refreshed on its own or the bar looks stalled between steps
    if current > last or processed % 500 == 0:
        emit_progress(max(current, last), f"{prefix} ({processed}/{_safe_total(total)} frames)")
        return max(current, last)
    return last


class _WindowedScorer:
    """Per-frame cut probabilities, windowed the way TransNetV2 expects.

    The model was trained on 100-frame windows in which only the middle 50
    predictions count - the 25 frames on either side are temporal context and
    nothing more. Scoring those context frames, as this module used to, invents
    cuts at the head of a video and weakens real ones at every window seam.

    Frames are pushed in decode order so callers can keep streaming instead of
    holding a whole episode in memory. The head is padded with copies of the
    first frame and the tail with copies of the last, matching
    ``TransNetV2.predict_frames``.

    Example:
        >>> scorer = _WindowedScorer(model, "cuda")
        >>> for frame in frames:
        ...     scorer.push(frame)
        >>> probs = scorer.finish()
    """

    def __init__(
        self,
        model,
        device: str,
        window_size: int = WINDOW_SIZE,
        stride: int = STRIDE,
        window_batch: int | None = None,
    ) -> None:
        self._model = model
        self._device = device
        self._window = window_size
        self._stride = stride
        self._pad = (window_size - stride) // 2
        self._window_batch = window_batch if window_batch is not None else (8 if device == "cuda" else 4)
        self._tail: np.ndarray = np.empty((0, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS), dtype=np.uint8)
        self._probs: list[np.ndarray] = []
        self._count = 0
        self._last: np.ndarray | None = None

    def push(self, frame: np.ndarray) -> None:
        """Feed one decoded frame of shape ``(27, 48, 3)``."""
        self.push_batch(frame[np.newaxis, ...])

    def push_batch(self, frames: np.ndarray) -> None:
        """Feed a batch of decoded frames of shape ``(K, 27, 48, 3)``."""
        if len(frames) == 0:
            return
        if self._count == 0:
            head_pad = np.repeat(frames[:1], self._pad, axis=0)
            self._tail = np.concatenate([head_pad, frames], axis=0)
        elif len(self._tail) > 0:
            self._tail = np.concatenate([self._tail, frames], axis=0)
        else:
            self._tail = frames
        self._last = frames[-1]
        self._count += len(frames)
        self._drain(flush=False)

    def finish(self) -> np.ndarray:
        """Pad the tail, score the remainder, return one probability per frame."""
        if self._count == 0 or self._last is None:
            return np.empty(0, dtype=np.float32)

        remainder = self._count % self._stride
        tail_pad = self._pad + self._stride - (remainder if remainder else self._stride)
        pad_arr = np.repeat(self._last[np.newaxis, ...], tail_pad, axis=0)
        self._tail = np.concatenate([self._tail, pad_arr], axis=0) if len(self._tail) > 0 else pad_arr
        self._drain(flush=True)

        probs = np.concatenate(self._probs) if self._probs else np.empty(0, dtype=np.float32)
        if len(probs) < self._count:
            probs = np.pad(probs, (0, self._count - len(probs)))
        return probs[: self._count].astype(np.float32, copy=False)

    def _score_windows(self, windows_arr: np.ndarray) -> None:
        import torch

        tensor = torch.from_numpy(windows_arr).to(self._device)
        with torch.inference_mode():
            logits, _ = self._model(tensor)
            preds = torch.sigmoid(logits)
        preds_np = preds.detach().cpu().numpy()
        if preds_np.ndim == 3 and preds_np.shape[-1] == 1:
            preds_np = preds_np[..., 0]
        elif preds_np.ndim == 1:
            preds_np = preds_np[None, :]
        self._probs.append(preds_np[:, self._pad : self._pad + self._stride].reshape(-1))

    def _drain(self, flush: bool = False) -> None:
        available = len(self._tail)
        if available < self._window:
            return
        num_windows = (available - self._window) // self._stride + 1
        if not flush and num_windows < self._window_batch:
            return
        total_consume = (num_windows - 1) * self._stride + self._window
        s0, s1, s2, s3 = self._tail.strides
        windows_view = np.lib.stride_tricks.as_strided(
            self._tail,
            shape=(num_windows, self._window, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS),
            strides=(s0 * self._stride, s0, s1, s2, s3),
        )
        for start in range(0, num_windows, self._window_batch):
            end = min(num_windows, start + self._window_batch)
            if not flush and (end - start) < self._window_batch:
                num_windows = start
                break
            self._score_windows(np.ascontiguousarray(windows_view[start:end]))
        if num_windows > 0:
            consumed_frames = num_windows * self._stride
            self._tail = np.ascontiguousarray(self._tail[consumed_frames:])


def _scores_to_scenes(model, scores: np.ndarray, threshold: float) -> np.ndarray:
    """Turn per-frame probabilities into ``(N, 2)`` frame ranges.

    ``predictions_to_scenes`` walks the array with a running index, so an empty
    one raises before it can return anything.
    """
    if len(scores) == 0:
        return np.empty((0, 2), dtype=np.int32)
    return model.predictions_to_scenes(scores, threshold=threshold)


def _transnet_ffmpeg_candidates(input_video: str | Path, codec: str | None, device: str) -> list[tuple[str, list[str]]]:
    """Build ordered ``(label, argv)`` FFmpeg decode commands with GPU acceleration and CPU fallback."""
    ffmpeg = get_ffmpeg()
    src = str(input_video)
    out_tail = ["-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    cpu_cmd = [
        ffmpeg, "-nostdin", "-y", "-flags2", "fast", "-i", src,
        "-an", "-sn", "-dn",
        "-vf", f"scale={FRAME_WIDTH}:{FRAME_HEIGHT}:flags=fast_bilinear",
        *out_tail,
    ]
    if os.environ.get("AMVERGE_SCOUT_HWACCEL", "").strip() == "0":
        return [("cpu", cpu_cmd)]

    candidates: list[tuple[str, list[str]]] = []
    if device == "cuda" and codec in NVDEC_CODECS and sys.platform in ("win32", "linux"):
        candidates.append((
            f"cuvid:{codec}",
            [
                ffmpeg, "-nostdin", "-y",
                "-hwaccel", "cuda", "-c:v", f"{codec}_cuvid", "-resize", "96x54",
                "-i", src, "-an", "-sn", "-dn",
                "-vf", f"scale={FRAME_WIDTH}:{FRAME_HEIGHT}:flags=fast_bilinear",
                *out_tail,
            ],
        ))
    elif sys.platform == "darwin":
        candidates.append((
            "videotoolbox",
            [
                ffmpeg, "-nostdin", "-y", "-hwaccel", "videotoolbox",
                "-i", src, "-an", "-sn", "-dn",
                "-vf", f"scale={FRAME_WIDTH}:{FRAME_HEIGHT}:flags=fast_bilinear",
                *out_tail,
            ],
        ))
    elif sys.platform.startswith("linux") and device != "cuda":
        candidates.append((
            "vaapi",
            [
                ffmpeg, "-nostdin", "-y",
                "-hwaccel", "vaapi", "-hwaccel_output_format", "vaapi",
                "-i", src, "-an", "-sn", "-dn",
                "-vf", f"scale_vaapi=w=96:h=54,hwdownload,scale={FRAME_WIDTH}:{FRAME_HEIGHT}:flags=fast_bilinear",
                *out_tail,
            ],
        ))

    candidates.append(("cpu", cpu_cmd))
    return candidates


def _open_ffmpeg_frame_stream(input_video: str | Path, codec: str | None, device: str, chunk_bytes: int):
    """Spawn FFmpeg using the fastest working hardware decoder, falling back to CPU on first-frame failure."""
    candidates = _transnet_ffmpeg_candidates(input_video, codec, device)
    for idx, (label, cmd) in enumerate(candidates):
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            bufsize=chunk_bytes * 2, creationflags=CREATE_NO_WINDOW,
        )
        if process.stdout is None:
            process.kill()
            process.wait()
            continue
        first_chunk = b""
        while len(first_chunk) < FRAME_BYTES:
            part = process.stdout.read(chunk_bytes - len(first_chunk))
            if not part:
                break
            first_chunk += part
        if len(first_chunk) >= FRAME_BYTES or idx == len(candidates) - 1:
            return label, process, first_chunk
        process.stdout.close()
        process.terminate()
        process.wait()
    raise RuntimeError("Failed to start FFmpeg decode stream")


def decode_and_detect_scenes(
    input_video: str | Path,
    threshold: float = DEFAULT_THRESHOLD,
    *,
    device: str | None = None,
    return_frames: bool = False,
):
    """Decode video frames and detect scenes with TransNetV2 in one streaming pass.

    Uses FFmpeg (with hardware-accelerated decode + downscale when available,
    falling back seamlessly to CPU) to pipe raw RGB frames (48x27) into a
    cached TransNetV2 model on a background reader thread.
    """
    if not TRANSNET_AVAILABLE:
        raise ImportError(
            "transnetv2_pytorch not installed. Run: pip install amverge[ml]"
        )

    emit_progress(10, "Calculating frame info...")
    video_fps = probe_video_fps(input_video)
    video_duration = probe_video_duration(input_video)
    total_frames = probe_video_total_frames(input_video, video_fps, video_duration)
    codec = probe_video_codec(input_video)

    emit_progress(20, "Loading TransNetV2 model...")
    import torch
    if device is None:
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    model = get_transnet_model(device)

    chunk_frames = 256
    chunk_bytes = chunk_frames * FRAME_BYTES
    hw_label, process, first_chunk = _open_ffmpeg_frame_stream(input_video, codec, device, chunk_bytes)
    log(f"[diag] TransNetV2 FFmpeg stream ({hw_label}, device={device})")

    chunk_queue: queue.Queue = queue.Queue(maxsize=16)

    def _reader() -> None:
        leftover = b""
        try:
            pending = first_chunk
            while True:
                if not pending:
                    pending = process.stdout.read(chunk_bytes)
                    if not pending:
                        break
                buf = leftover + pending
                pending = b""
                usable = (len(buf) // FRAME_BYTES) * FRAME_BYTES
                leftover = buf[usable:]
                if usable == 0:
                    continue
                arr = np.frombuffer(buf[:usable], dtype=np.uint8).reshape(
                    -1, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS
                ).copy()
                chunk_queue.put(arr)
        except BaseException as exc:
            chunk_queue.put(exc)
        finally:
            chunk_queue.put(None)

    reader_thread = threading.Thread(target=_reader, name="transnet-ffmpeg-reader", daemon=True)
    reader_thread.start()

    scorer = _WindowedScorer(model, device)
    processed = 0
    last_progress = 19
    collected: list[np.ndarray] = []

    try:
        while True:
            item = chunk_queue.get()
            if item is None:
                break
            if isinstance(item, BaseException):
                raise item
            if return_frames:
                collected.append(item)
            scorer.push_batch(item)
            processed += len(item)
            last_progress = _emit_loop_progress(
                processed, total_frames, 20, 30, "Decoding video (FFmpeg)...", last_progress
            )
    finally:
        reader_thread.join(timeout=5.0)
        if process.stdout:
            process.stdout.close()
        process.wait()

    emit_progress(50, f"Decoding video (FFmpeg)... ({processed}/{_safe_total(total_frames)})")

    scores_arr = scorer.finish()
    scenes_frames = _scores_to_scenes(model, scores_arr, threshold)
    scenes_secs = transnet_scenes_to_seconds(scenes_frames, video_fps)

    if return_frames:
        frames_arr = (
            np.empty((0, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS), dtype=np.uint8)
            if not collected
            else (collected[0] if len(collected) == 1 else np.concatenate(collected, axis=0))
        )
        return scenes_secs, scenes_frames, frames_arr
    return scenes_secs, scenes_frames


def decode_video_frames_ffmpeg(input_video: str | Path) -> np.ndarray:
    """Decode all video frames into a numpy array using FFmpeg (cross-platform).

    Pipes raw RGB24 frames at TransNetV2 input resolution (48x27) from an
    FFmpeg subprocess. Used wherever Nelux (Windows-only NVDEC decode) is
    unavailable, e.g. on Linux/macOS or a machine with no NVIDIA GPU. Mirrors
    :func:`decode_video_frames_nelux`'s return contract so callers can swap
    between the two decode paths freely.

    Args:
        input_video: Path to the source video file.

    Returns:
        ndarray of shape ``(num_frames, 27, 48, 3)`` with dtype ``uint8``.

    Example:
        >>> frames = decode_video_frames_ffmpeg("episode.mp4")
        >>> print(frames.shape)  # (378, 27, 48, 3)
    """
    log("Running FFmpeg video decode...")

    from ..infra.binaries import get_ffmpeg

    video_fps = probe_video_fps(input_video)
    video_duration = probe_video_duration(input_video)
    total_frames = probe_video_total_frames(input_video, video_fps, video_duration)

    cmd = [
        get_ffmpeg(), "-y",
        "-flags2", "fast",
        "-i", str(input_video),
        "-pix_fmt", "rgb24",
        "-vf", f"scale={FRAME_WIDTH}:{FRAME_HEIGHT}:flags=fast_bilinear",
        "-f", "rawvideo",
        "pipe:1",
    ]
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW
    )
    if process.stdout is None:
        raise RuntimeError("Failed to create stdout pipe")

    chunks: list[np.ndarray] = []
    actual_frames = 0
    last_progress = 19
    chunk_bytes = 256 * FRAME_BYTES
    leftover = b""

    while True:
        chunk = process.stdout.read(chunk_bytes)
        if not chunk:
            break
        buf = leftover + chunk
        usable = (len(buf) // FRAME_BYTES) * FRAME_BYTES
        leftover = buf[usable:]
        if usable == 0:
            continue
        arr = np.frombuffer(buf[:usable], dtype=np.uint8).reshape(
            -1, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS
        ).copy()
        chunks.append(arr)
        actual_frames += len(arr)
        last_progress = _emit_loop_progress(
            actual_frames, total_frames, 20, 35, "Decoding video (FFmpeg)...", last_progress
        )

    process.stdout.close()
    process.wait()

    emit_progress(55, f"Decoding video (FFmpeg)... ({actual_frames}/{_safe_total(total_frames)})")

    if not chunks:
        return np.empty((0, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS), dtype=np.uint8)

    return chunks[0] if len(chunks) == 1 else np.concatenate(chunks, axis=0)


def decode_video_frames_nelux(input_video: str | Path) -> np.ndarray:
    """Decode all video frames into a numpy array using Nelux (Windows only).

    Reads every frame at TransNetV2 input resolution (48x27 RGB).
    Uses NVDEC hardware acceleration when CUDA is available.

    Args:
        input_video: Path to the source video file.

    Returns:
        ndarray of shape ``(num_frames, 27, 48, 3)`` with dtype ``uint8``.

    Raises:
        ImportError: If Nelux is not installed, or is installed beside a
            PyTorch version it will not load with.
        RuntimeError: If NVDEC cannot decode this particular file. 10-bit H.264
            is the common case, and the hardware simply does not support it, so
            callers should fall back to :func:`decode_video_frames_ffmpeg`. Codecs
            outside :data:`NVDEC_CODECS` raise up front: NVDEC hands back frames for
            MPEG-4 Part 2 (XviD) and MPEG-2 that are unusable, so TransNetV2 sees a
            single scene. Also raised when Nelux stops returning frames for
            :data:`NELUX_STALL_SECONDS`: its native ``read_frame`` deadlocks at
            random on some long 10-bit HEVC files (3 runs in 5 on one episode),
            so reads run on a daemon thread the caller can walk away from.
            Nelux 0.21 fixed that deadlock; it also made ``len(reader)`` an
            exact count that decodes the whole file first, so the reader opens
            with ``seek_mode="approximate"`` (metadata count). 0.18 has no such
            argument and raises TypeError, which falls back to the plain call.

    Example:
        >>> frames = decode_video_frames_nelux("episode.mp4")
        >>> print(frames.shape)  # (378, 27, 48, 3)
    """

    import torch

    codec = probe_video_codec(input_video)
    if codec not in NVDEC_CODECS:
        raise RuntimeError(f"NVDEC does not decode {codec or 'this video'} reliably")

    VideoReader = _get_nelux_video_reader()
    # labelled separately because Nelux silently drops to CPU without CUDA
    decode_accelerator = "nvdec" if torch.cuda.is_available() else "cpu"
    label = "Nelux NVDEC" if decode_accelerator == "nvdec" else "Nelux CPU"

    log(f"Running {label} video decode...")
    options = {"decode_accelerator": decode_accelerator, "resize": (FRAME_WIDTH, FRAME_HEIGHT)}
    try:
        reader = VideoReader(str(input_video), seek_mode="approximate", **options)
    except TypeError:
        reader = VideoReader(str(input_video), **options)

    total_frames = len(reader)
    frames = np.empty(
        (total_frames, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS),
        dtype=np.uint8,
    )

    state = {"count": 0, "error": None, "done": False}

    def read_all() -> None:
        try:
            for i in range(total_frames):
                frame = reader.read_frame()
                if frame is None:
                    break
                frames[i] = _nelux_frame_to_array(frame, torch)
                state["count"] = i + 1
        except BaseException as exc:
            state["error"] = exc
        finally:
            state["done"] = True

    worker = threading.Thread(target=read_all, name="nelux-decode", daemon=True)
    worker.start()

    last_count, last_change, last_progress = 0, time.monotonic(), 19
    while not state["done"]:
        worker.join(timeout=0.5)
        count = state["count"]
        if count != last_count:
            last_count, last_change = count, time.monotonic()
            last_progress = max(last_progress, int(20 + 35 * count / _safe_total(total_frames)))
            emit_progress(last_progress, f"Decoding video ({label})... ({count}/{_safe_total(total_frames)} frames)")
        elif not state["done"] and time.monotonic() - last_change > NELUX_STALL_SECONDS:
            raise RuntimeError(
                f"Nelux stopped returning frames at {count}/{total_frames} "
                f"(no frame for {NELUX_STALL_SECONDS:.0f}s)"
            )

    if state["error"] is not None:
        raise state["error"]

    actual_frames = state["count"]
    if actual_frames < total_frames:
        frames = frames[:actual_frames]

    emit_progress(55, f"Decoding video ({label})... ({actual_frames}/{_safe_total(total_frames)})")
    return frames


def _nelux_frame_to_array(frame, torch) -> np.ndarray:
    """One Nelux frame as a ``(27, 48, 3)`` uint8 array."""
    if isinstance(frame, torch.Tensor):
        frame_np = frame.detach().to("cpu").numpy()
    else:
        frame_np = np.asarray(frame)

    # sources deeper than 8-bit come back as full-range uint16 (0-65535). a
    # plain uint8 cast keeps only the low byte, which turned 10-bit decodes
    # into noise; the high byte is the 8-bit value FFmpeg would give
    if frame_np.dtype == np.uint16:
        frame_np = frame_np >> 8
    frame_np = frame_np.astype(np.uint8, copy=False)

    if frame_np.ndim != 3:
        raise ValueError(f"Unexpected frame rank from nelux: {frame_np.ndim}")

    if frame_np.shape[0] == FRAME_CHANNELS and frame_np.shape[-1] != FRAME_CHANNELS:
        frame_np = np.transpose(frame_np, (1, 2, 0))

    if frame_np.shape != (FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS):
        raise ValueError(
            f"Unexpected frame shape from nelux. Got {frame_np.shape}, "
            f"expected ({FRAME_HEIGHT}, {FRAME_WIDTH}, {FRAME_CHANNELS})."
        )

    return frame_np


def run_model_one_pass(
    frames: np.ndarray,
    input_file: str | Path,
    batch_size: int = 100,
    overlap: int = 50,
    device: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Run TransNetV2 inference on pre-decoded frames.

    Splits the frame array into overlapping windows of ``batch_size`` frames
    (default 100) and keeps the middle ``batch_size - overlap`` predictions of
    each - the frames on either edge are context only. GPU-accelerated when
    CUDA is available, MPS if MPS available.

    Args:
        frames: Frame array of shape ``(N, 27, 48, 3)`` with dtype ``uint8``.
        input_file: Path to the source video (used for FPS probe).
        device: Torch device to run inference on. Auto-detected (cuda >
            mps > cpu) when not given.
        batch_size: Number of frames per inference batch.
        overlap: Overlap between consecutive batches (default 50 frames).
        threshold: Cut confidence in ``0-1`` a frame must exceed to end a scene.
            Lower cuts more.

    Returns:
        Tuple of ``(scenes_secs, scenes_frames)`` - both ``(N, 2)`` ndarrays.

    Raises:
        ImportError: If ``transnetv2_pytorch`` is not installed.
            Run ``pip install amverge[ml]``.

    Example:
        >>> from amverge.core.detection.ai_scene_detection import (
        ...     decode_video_frames_nelux, run_model_one_pass
        ... )
        >>> frames = decode_video_frames_nelux("episode.mp4")
        >>> secs, frm = run_model_one_pass(frames, "episode.mp4")
        >>> print(f"{len(secs)} scenes detected")
    """
    if not TRANSNET_AVAILABLE:
        raise ImportError(
            "transnetv2_pytorch not installed. Run: pip install amverge[ml]"
        )

    log("Running TransNetV2 one-pass inference...")
    import torch
    num_frames = len(frames)
    stride = batch_size - overlap

    if device is None:
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    model = get_transnet_model(device)
    video_fps = probe_video_fps(input_file)

    scorer = _WindowedScorer(model, device, batch_size, stride)
    chunk_size = stride * 8
    last_progress = 54
    for start in range(0, num_frames, chunk_size):
        end = min(num_frames, start + chunk_size)
        scorer.push_batch(frames[start:end])
        last_progress = _emit_loop_progress(
            end, num_frames, 55, 20,
            "Running TransNetV2 scene detection...", last_progress,
        )

    final_scores = scorer.finish()
    scenes_frames = _scores_to_scenes(model, final_scores, threshold)
    emit_progress(75, f"TransNetV2 complete ({num_frames}/{_safe_total(num_frames)} frames)")

    scenes_secs = transnet_scenes_to_seconds(scenes_frames, video_fps)
    return scenes_secs, scenes_frames


def detect_transnet_with_fallback(
    video_path: str | Path,
    *,
    use_nelux: bool = False,
    threshold: float = DEFAULT_THRESHOLD,
    on_nelux_fallback: Callable[[Exception], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if use_nelux:
        try:
            frames = decode_video_frames_nelux(video_path)
            return run_model_one_pass(frames, video_path, threshold=threshold)
        except Exception as exc:
            if on_nelux_fallback is not None:
                on_nelux_fallback(exc)
    return decode_and_detect_scenes(video_path, threshold=threshold)
