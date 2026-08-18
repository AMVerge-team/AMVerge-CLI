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
import subprocess
import sys
from pathlib import Path

import numpy as np

from ..infra.ipc import emit_progress, log
from .nelux_runtime import _get_nelux_video_reader
from ..video.probe_utils import probe_video_fps, probe_video_duration, probe_video_total_frames

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
from ..video.scene_utils import scenes_frames_to_seconds
from ..transnet.transnet_constants import (
    FRAME_BYTES,
    FRAME_CHANNELS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    WINDOW_SIZE,
    STRIDE,
)

try:
    from transnetv2_pytorch import TransNetV2 as _TransNetV2
    TRANSNET_AVAILABLE = True
except ImportError:
    TRANSNET_AVAILABLE = False

DEFAULT_THRESHOLD = 0.5


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
    if current > last:
        emit_progress(current, f"{prefix} ({processed}/{_safe_total(total)} frames)")
        return current
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
    ) -> None:
        self._model = model
        self._device = device
        self._window = window_size
        self._stride = stride
        self._pad = (window_size - stride) // 2
        self._buffer: list[np.ndarray] = []
        self._probs: list[float] = []
        self._count = 0
        self._last: np.ndarray | None = None

    def push(self, frame: np.ndarray) -> None:
        """Feed one decoded frame of shape ``(27, 48, 3)``."""
        if self._count == 0:
            self._buffer.extend([frame] * self._pad)
        self._buffer.append(frame)
        self._last = frame
        self._count += 1
        self._drain()

    def finish(self) -> np.ndarray:
        """Pad the tail, score the remainder, return one probability per frame."""
        if self._count == 0 or self._last is None:
            return np.empty(0, dtype=np.float32)

        remainder = self._count % self._stride
        tail_pad = self._pad + self._stride - (remainder if remainder else self._stride)
        self._buffer.extend([self._last] * tail_pad)
        self._drain()

        probs = np.asarray(self._probs, dtype=np.float32)
        if len(probs) < self._count:
            probs = np.pad(probs, (0, self._count - len(probs)))
        return probs[: self._count]

    def _drain(self) -> None:
        import torch
        while len(self._buffer) >= self._window:
            batch = np.stack(self._buffer[: self._window])
            tensor = torch.from_numpy(batch).unsqueeze(dim=0).to(self._device)
            with torch.inference_mode():
                logits, _ = self._model(tensor)
                preds = torch.sigmoid(logits)
            preds = preds.detach().cpu().numpy().squeeze()
            self._probs.extend(preds[self._pad : self._pad + self._stride].tolist())
            self._buffer = self._buffer[self._stride :]


def _scores_to_scenes(model, scores: np.ndarray, threshold: float) -> np.ndarray:
    """Turn per-frame probabilities into ``(N, 2)`` frame ranges.

    ``predictions_to_scenes`` walks the array with a running index, so an empty
    one raises before it can return anything.
    """
    if len(scores) == 0:
        return np.empty((0, 2), dtype=np.int32)
    return model.predictions_to_scenes(scores, threshold=threshold)


def decode_and_detect_scenes(
    input_video: str | Path,
    threshold: float = DEFAULT_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode video frames and detect scenes with TransNetV2 in one call.

    Uses FFmpeg to pipe raw RGB frames (48x27) into a TransNetV2 model.
    Works cross-platform without Nelux DLLs.

    Args:
        input_video: Path to the source video file.
        threshold: Cut confidence in ``0-1`` a frame must exceed to end a scene.
            Lower cuts more.

    Returns:
        Tuple of ``(scenes_secs, scenes_frames)`` - both are ``(N, 2)``
        ndarrays where each row is ``[start, end]`` in seconds or frames.

    Raises:
        ImportError: If ``transnetv2_pytorch`` is not installed.
            Run ``pip install amverge[ml]``.

    Example:
        >>> scenes_secs, scenes_frames = decode_and_detect_scenes("ep.mp4")
        >>> for start, end in scenes_secs:
        ...     print(f"Scene: {start:.1f}s - {end:.1f}s")
    """
    if not TRANSNET_AVAILABLE:
        raise ImportError(
            "transnetv2_pytorch not installed. Run: pip install amverge[ml]"
        )

    from transnetv2_pytorch import TransNetV2

    emit_progress(10, "Calculating frame info...")
    video_fps = probe_video_fps(input_video)
    video_duration = probe_video_duration(input_video)
    total_frames = probe_video_total_frames(input_video, video_fps, video_duration)

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_video),
        "-pix_fmt", "rgb24",
        "-vf", "scale=48:27",
        "-f", "rawvideo",
        "pipe:1",
    ]
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW
    )
    if process.stdout is None:
        raise RuntimeError("Failed to create stdout pipe")

    emit_progress(20, "Loading TransNetV2 model...")
    import torch
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    with contextlib.redirect_stdout(sys.stderr):
        model = TransNetV2(device=device)
    model.eval()

    scorer = _WindowedScorer(model, device)
    processed = 0
    last_progress = 19

    while True:
        raw_frame = process.stdout.read(FRAME_BYTES)
        if len(raw_frame) == 0:
            break
        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
            FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS
        )
        scorer.push(frame)

        processed += 1
        if processed % 10 == 0:
            last_progress = _emit_loop_progress(
                processed, total_frames, 20, 30, "Decoding video...", last_progress
            )

    process.stdout.close()
    process.wait()

    emit_progress(50, f"Decoding video... ({processed}/{_safe_total(total_frames)})")

    scores_arr = scorer.finish()
    scenes_frames = _scores_to_scenes(model, scores_arr, threshold)
    scenes_secs = scenes_frames_to_seconds(scenes_frames, video_fps)

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
        "-i", str(input_video),
        "-pix_fmt", "rgb24",
        "-vf", f"scale={FRAME_WIDTH}:{FRAME_HEIGHT}",
        "-f", "rawvideo",
        "pipe:1",
    ]
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW
    )
    if process.stdout is None:
        raise RuntimeError("Failed to create stdout pipe")

    frames: list[np.ndarray] = []
    actual_frames = 0
    last_progress = 19

    while True:
        raw_frame = process.stdout.read(FRAME_BYTES)
        if len(raw_frame) == 0:
            break
        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
            FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS
        )
        frames.append(frame)
        actual_frames += 1
        if actual_frames % 10 == 0:
            last_progress = _emit_loop_progress(
                actual_frames, total_frames, 20, 35, "Decoding video...", last_progress
            )

    process.stdout.close()
    process.wait()

    emit_progress(55, f"Decoding video... ({actual_frames}/{_safe_total(total_frames)})")

    if not frames:
        return np.empty((0, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS), dtype=np.uint8)

    return np.stack(frames)


def decode_video_frames_nelux(input_video: str | Path) -> np.ndarray:
    """Decode all video frames into a numpy array using Nelux (Windows only).

    Reads every frame at TransNetV2 input resolution (48x27 RGB).
    Uses NVDEC hardware acceleration when CUDA is available.

    Args:
        input_video: Path to the source video file.

    Returns:
        ndarray of shape ``(num_frames, 27, 48, 3)`` with dtype ``uint8``.

    Raises:
        ImportError: If Nelux is not installed or FFmpeg DLLs are not found.
            Set ``AMVERGE_FFMPEG_BIN`` to the directory containing the DLLs.

    Example:
        >>> frames = decode_video_frames_nelux("episode.mp4")
        >>> print(frames.shape)  # (378, 27, 48, 3)
    """

    log("Running nelux video decode...")
    import torch
    VideoReader = _get_nelux_video_reader()
    decode_accelerator = "nvdec" if torch.cuda.is_available() else "cpu"
    reader = VideoReader(
        str(input_video),
        decode_accelerator=decode_accelerator,
        resize=(FRAME_WIDTH, FRAME_HEIGHT),
    )

    total_frames = len(reader)
    frames = np.empty(
        (total_frames, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS),
        dtype=np.uint8,
    )

    actual_frames = 0
    last_progress = 19

    for i in range(total_frames):
        frame = reader.read_frame()
        if frame is None:
            break

        if isinstance(frame, torch.Tensor):
            frame_np = frame.detach().to("cpu").numpy().astype(np.uint8, copy=False)
        else:
            frame_np = np.asarray(frame, dtype=np.uint8)

        if frame_np.ndim != 3:
            raise ValueError(f"Unexpected frame rank from nelux: {frame_np.ndim}")

        if frame_np.shape[0] == FRAME_CHANNELS and frame_np.shape[-1] != FRAME_CHANNELS:
            frame_np = np.transpose(frame_np, (1, 2, 0))

        if frame_np.shape != (FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS):
            raise ValueError(
                f"Unexpected frame shape from nelux. Got {frame_np.shape}, "
                f"expected ({FRAME_HEIGHT}, {FRAME_WIDTH}, {FRAME_CHANNELS})."
            )

        frames[i] = frame_np
        actual_frames += 1
        if actual_frames % 10 == 0:
            last_progress = _emit_loop_progress(
                actual_frames, total_frames, 20, 35, "Decoding video...", last_progress
            )

    if actual_frames < total_frames:
        frames = frames[:actual_frames]

    emit_progress(55, f"Decoding video... ({actual_frames}/{_safe_total(total_frames)})")
    return frames


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

    from transnetv2_pytorch import TransNetV2

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

    with contextlib.redirect_stdout(sys.stderr):
        model = TransNetV2(device=device)
    model.eval()
    video_fps = probe_video_fps(input_file)

    last_progress = 54

    scorer = _WindowedScorer(model, device, batch_size, stride)

    for index in range(num_frames):
        scorer.push(frames[index])
        if (index + 1) % stride == 0:
            last_progress = _emit_loop_progress(
                index + 1, num_frames, 55, 20,
                "Running TransNetV2 scene detection...", last_progress,
            )

    final_scores = scorer.finish()
    scenes_frames = _scores_to_scenes(model, final_scores, threshold)
    emit_progress(75, f"TransNetV2 complete ({num_frames}/{_safe_total(num_frames)} frames)")

    scenes_secs = scenes_frames_to_seconds(scenes_frames, video_fps)
    return scenes_secs, scenes_frames
