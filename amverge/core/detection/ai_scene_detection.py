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

import subprocess
import sys
from pathlib import Path

import numpy as np

from ..infra.ipc import emit_progress, log
from .nelux_runtime import _get_nelux_video_reader
from ..video.probe_utils import probe_video_fps, probe_video_duration, probe_video_total_frames
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


def _run_model_batch(
    model,
    batch: np.ndarray,
    start_index: int,
    scores: np.ndarray,
    counts: np.ndarray,
    device: str,
) -> None:
    import torch
    tensor = torch.from_numpy(batch).unsqueeze(dim=0).to(device)
    with torch.inference_mode():
        single_frame_pred, _ = model(tensor)
    preds = single_frame_pred.detach().cpu().numpy().squeeze()
    end = len(batch)
    for i, pred in enumerate(preds):
        global_index = start_index + i
        scores[global_index] += pred
        counts[global_index] += 1


def decode_and_detect_scenes(
    input_video: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode video frames and detect scenes with TransNetV2 in one call.

    Uses FFmpeg to pipe raw RGB frames (48x27) into a TransNetV2 model.
    Works cross-platform without Nelux DLLs.

    Args:
        input_video: Path to the source video file.

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
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if process.stdout is None:
        raise RuntimeError("Failed to create stdout pipe")

    emit_progress(20, "Loading TransNetV2 model...")
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransNetV2(device=device)
    model.eval()

    window_start_index = 0
    buffer: list[np.ndarray] = []
    scores = []
    counts = []

    processed = 0
    last_progress = 19

    while True:
        raw_frame = process.stdout.read(FRAME_BYTES)
        if len(raw_frame) == 0:
            break
        frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
            FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS
        )
        buffer.append(frame)
        scores.append(0.0)
        counts.append(0)

        if len(buffer) >= WINDOW_SIZE:
            batch = np.stack(buffer[:WINDOW_SIZE])
            _run_model_batch(model, batch, window_start_index, scores, counts, device)
            buffer = buffer[STRIDE:]
            window_start_index += STRIDE

        processed += 1
        if processed % 10 == 0:
            last_progress = _emit_loop_progress(
                processed, total_frames, 20, 30, "Decoding video...", last_progress
            )

    if buffer:
        batch = np.stack(buffer)
        _run_model_batch(model, batch, window_start_index, scores, counts, device)

    emit_progress(50, f"Decoding video... ({processed}/{_safe_total(total_frames)})")

    scores_arr = np.array(scores) / np.array(counts)
    scenes_frames = model.predictions_to_scenes(scores_arr)
    scenes_secs = scenes_frames_to_seconds(scenes_frames, video_fps)

    return scenes_secs, scenes_frames


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
) -> tuple[np.ndarray, np.ndarray]:
    """Run TransNetV2 inference on pre-decoded frames.

    Splits the frame array into overlapping windows of ``batch_size`` frames
    (default 100), runs the model on each batch, and averages overlapping
    predictions. GPU-accelerated when CUDA is available.

    Args:
        frames: Frame array of shape ``(N, 27, 48, 3)`` with dtype ``uint8``.
        input_file: Path to the source video (used for FPS probe).
        batch_size: Number of frames per inference batch.
        overlap: Overlap between consecutive batches (default 50 frames).

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
    scores = np.zeros(num_frames)
    counts = np.zeros(num_frames)
    stride = batch_size - overlap

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TransNetV2(device=device)
    model.eval()
    video_fps = probe_video_fps(input_file)

    last_progress = 54

    for start in range(0, num_frames, stride):
        end = min(start + batch_size, num_frames)
        frames_batch = frames[start:end].copy()

        tensor = torch.from_numpy(frames_batch).unsqueeze(dim=0).to(device)
        with torch.inference_mode():
            single_frame_pred, _ = model(tensor)
        preds = single_frame_pred.detach().cpu().numpy().squeeze()

        scores[start:end] += preds
        counts[start:end] += 1

        last_progress = _emit_loop_progress(
            end, num_frames, 55, 20, "Running TransNetV2 scene detection...", last_progress
        )

    final_scores = scores / counts
    scenes_frames = model.predictions_to_scenes(final_scores)
    emit_progress(75, f"TransNetV2 complete ({num_frames}/{_safe_total(num_frames)} frames)")

    scenes_secs = scenes_frames_to_seconds(scenes_frames, video_fps)
    return scenes_secs, scenes_frames
