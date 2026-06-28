from __future__ import annotations

"""Smart scene cutting with lossless copy, smartcut, and re-encode fallback.

Four cut modes depending on scene boundary alignment with keyframes:

    copy         - start is on a keyframe: lossless stream copy
    snapped_copy - HEVC CPU, nearest keyframe within 5s: lossless from snap
    smartcut     - H.264, next keyframe within 90%: encode head + lossless tail
    reencode     - fallback: full re-encode with NVENC (GPU) or CPU encoder

Usage:
    >>> from amverge.core.smart_cut import cut_all_scenes
    >>> from amverge.core.keyframe_align import get_keyframe_timestamps_pyav
    >>> from amverge.core.codec_utils import check_if_hevc
    >>> from pathlib import Path
    >>>
    >>> keyframes = get_keyframe_timestamps_pyav("episode.mp4")
    >>> is_hevc = check_if_hevc("episode.mp4")
    >>> scenes = [{"scene_index": 0, "start_sec": 0.0, "end_sec": 5.0}]
    >>>
    >>> results = cut_all_scenes(
    ...     input_file=Path("episode.mp4"),
    ...     scenes=scenes,
    ...     keyframes=keyframes,
    ...     out_dir=Path("./scenes"),
    ...     use_cuda=True,
    ...     is_hevc=is_hevc,
    ...     on_ready=lambda r: print(r["clip_mode"]),
    ... )
    >>> for r in results:
    ...     print(f"Scene {r['scene_index']}: {r['clip_mode']}")
"""

import os
import subprocess
from bisect import bisect_left, bisect_right
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from .binaries import get_ffmpeg
from .ipc import emit_progress, log

KEYFRAME_SNAP_THRESHOLD = 0.2
PRE_SEEK_OFFSET = 10.0
HEVC_SNAP_MAX = 5.0


def _background_kwargs() -> dict:
    if os.name == "nt":
        return {
            "creationflags": subprocess.BELOW_NORMAL_PRIORITY_CLASS | 0x08000000
        }
    return {}


def _run_ffmpeg(cmd: list[str]) -> None:
    p = subprocess.run(
        cmd, capture_output=True, text=True, timeout=120, **_background_kwargs()
    )
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed (exit {p.returncode}): {p.stderr[-600:]}")


def _lossless_copy(
    input_file: Path,
    start: float,
    end: float,
    out_path: Path,
    *,
    aac_audio: bool = False,
) -> None:
    audio_args = ["-c:a", "aac", "-b:a", "128k"] if aac_audio else ["-c:a", "copy"]
    _run_ffmpeg([
        get_ffmpeg(), "-y",
        "-ss", f"{start:.3f}",
        "-i", str(input_file),
        "-t", f"{end - start:.3f}",
        "-map", "0:v:0", "-map", "0:a?",
        "-c:v", "copy",
        *audio_args,
        "-movflags", "+faststart",
        str(out_path),
    ])


def _encode_segment(
    input_file: Path, start: float, end: float, out_path: Path, use_cuda: bool
) -> None:
    pre_seek = max(0.0, start - PRE_SEEK_OFFSET)
    post_seek = start - pre_seek
    duration = end - start

    if use_cuda:
        encode_args = ["-c:v", "h264_nvenc", "-preset", "p1", "-rc", "vbr", "-cq", "16", "-b:v", "0"]
    else:
        encode_args = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "16"]

    cmd = [get_ffmpeg(), "-y"]
    if pre_seek > 0.0:
        cmd += ["-ss", f"{pre_seek:.3f}"]
    cmd += ["-i", str(input_file)]
    cmd += ["-ss", f"{post_seek:.3f}", "-t", f"{duration:.3f}"]
    cmd += ["-map", "0:v:0", "-map", "0:a?", "-pix_fmt", "yuv420p"]
    cmd += encode_args
    cmd += ["-c:a", "aac", "-b:a", "128k", str(out_path)]
    _run_ffmpeg(cmd)


def _concat_two(
    head_path: Path, tail_path: Path, out_path: Path, tmp_dir: Path, scene_idx: int
) -> None:
    list_file = tmp_dir / f"_concat_{scene_idx:04d}.txt"
    list_file.write_text(
        f"file '{head_path.as_posix()}'\nfile '{tail_path.as_posix()}'\n",
        encoding="utf-8",
    )
    try:
        _run_ffmpeg([
            get_ffmpeg(), "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            "-movflags", "+faststart",
            str(out_path),
        ])
    finally:
        list_file.unlink(missing_ok=True)


def _start_is_on_keyframe(start_sec: float, keyframes: list[float]) -> bool:
    i = bisect_left(keyframes, start_sec)
    for ci in (i - 1, i):
        if 0 <= ci < len(keyframes):
            if abs(keyframes[ci] - start_sec) <= KEYFRAME_SNAP_THRESHOLD:
                return True
    return False


def _find_next_keyframe_after(keyframes: list[float], after: float) -> float | None:
    i = bisect_right(keyframes, after)
    return keyframes[i] if i < len(keyframes) else None


def cut_scene(
    input_file: Path,
    start_sec: float,
    end_sec: float,
    scene_idx: int,
    out_dir: Path,
    keyframes: list[float],
    use_cuda: bool,
    is_hevc: bool,
) -> tuple[str, str]:
    """Cut a single scene using the best available method.

    Chooses cut mode automatically:
    - ``copy`` if start aligns with a keyframe (within 0.2s).
    - ``snapped_copy`` for HEVC on CPU if a keyframe exists within 5s.
    - ``smartcut`` for H.264 when next keyframe is within 90% of the scene.
    - ``reencode`` as fallback.

    Args:
        input_file: Source video file.
        start_sec: Scene start time in seconds.
        end_sec: Scene end time in seconds.
        scene_idx: Scene number, used for output filename ``scene_{idx:04d}.mp4``.
        out_dir: Directory for output clips.
        keyframes: Sorted list of keyframe timestamps.
        use_cuda: If True, use NVENC for re-encode (GPU). CPU fallback if
            encoder not available.
        is_hevc: Whether the source video is HEVC-encoded.

    Returns:
        Tuple of ``(clip_path, mode)`` where ``mode`` is one of
        ``"copy"``, ``"snapped_copy"``, ``"smartcut"``, or ``"reencode"``.

    Raises:
        ValueError: If ``start_sec >= end_sec`` (non-positive duration).
    """
    out_path = out_dir / f"scene_{scene_idx:04d}.mp4"
    duration = end_sec - start_sec

    if duration <= 0:
        raise ValueError(f"Non-positive duration for scene {scene_idx}: {duration:.3f}s")

    if _start_is_on_keyframe(start_sec, keyframes):
        _lossless_copy(input_file, start_sec, end_sec, out_path)
        return str(out_path), "copy"

    k_next = _find_next_keyframe_after(keyframes, start_sec)
    head_fraction = (k_next - start_sec) / duration if k_next is not None else 1.0

    if is_hevc and not use_cuda:
        i = bisect_right(keyframes, start_sec)
        snap_kf = None
        best_diff = float("inf")
        for ci in (i - 1, i):
            if 0 <= ci < len(keyframes):
                diff = abs(keyframes[ci] - start_sec)
                if diff < best_diff:
                    best_diff = diff
                    snap_kf = keyframes[ci]
        if snap_kf is not None and best_diff <= HEVC_SNAP_MAX and snap_kf < end_sec:
            _lossless_copy(input_file, snap_kf, end_sec, out_path)
            return str(out_path), "snapped_copy"

    can_smartcut = (
        not is_hevc
        and k_next is not None
        and k_next < end_sec
        and head_fraction < 0.9
    )

    if can_smartcut:
        head_path = out_dir / f"_head_{scene_idx:04d}.mp4"
        tail_path = out_dir / f"_tail_{scene_idx:04d}.mp4"
        try:
            _encode_segment(input_file, start_sec, k_next, head_path, use_cuda)
            _lossless_copy(input_file, k_next, end_sec, tail_path, aac_audio=True)
            _concat_two(head_path, tail_path, out_path, out_dir, scene_idx)
        finally:
            head_path.unlink(missing_ok=True)
            tail_path.unlink(missing_ok=True)
        return str(out_path), "smartcut"

    _encode_segment(input_file, start_sec, end_sec, out_path, use_cuda)
    return str(out_path), "reencode"


def cut_all_scenes(
    input_file: Path,
    scenes: list[dict],
    keyframes: list[float],
    out_dir: Path,
    use_cuda: bool,
    is_hevc: bool,
    max_workers: int = 4,
    on_ready: Callable[[dict], None] | None = None,
    progress_range: tuple[int, int] = (82, 97),
    emit_progress_updates: bool = True,
) -> list[dict]:
    """Cut multiple scenes in parallel using a thread pool.

    Each scene dict must have ``"scene_index"``, ``"start_sec"``,
    and ``"end_sec"`` keys. Cutting happens via :func:`cut_scene`.

    Args:
        input_file: Source video file.
        scenes: List of scene dicts with ``scene_index``, ``start_sec``,
            ``end_sec``.
        keyframes: Sorted keyframe timestamps from
            :func:`~amverge.core.keyframe_align.get_keyframe_timestamps_pyav`.
        out_dir: Directory for output clips.
        use_cuda: Enable NVENC GPU encode for re-encode fallback.
        is_hevc: Source video codec. Enables HEVC snapped-copy path.
        max_workers: Thread pool size. Phase 1 uses 8, Phase 2 uses 2.
        on_ready: Called per completed scene with
            ``{"scene_index": int, "clip_path": str, "clip_mode": str}``.
        progress_range: ``(start, end)`` tuple for IPC progress percentage
            during cutting.
        emit_progress_updates: If True, emit ``PROGRESS|pct|msg`` IPC events.

    Returns:
        List of result dicts, each containing ``scene_index``,
        ``clip_path``, and ``clip_mode``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(scenes)
    results: list[dict] = []
    if total == 0:
        return results

    def _cut_one(scene: dict) -> dict:
        idx = scene["scene_index"]
        try:
            clip_path, clip_mode = cut_scene(
                input_file,
                float(scene["start_sec"]),
                float(scene["end_sec"]),
                idx,
                out_dir,
                keyframes,
                use_cuda,
                is_hevc,
            )
            log(f"Scene {idx}: {clip_mode} -> {Path(clip_path).name}")
            return {"scene_index": idx, "clip_path": clip_path, "clip_mode": clip_mode}
        except Exception as exc:
            log(f"Warning: scene {idx} failed: {exc}")
            return {"scene_index": idx, "clip_path": None, "clip_mode": "failed"}

    workers = min(max_workers, max(1, total))
    completed = 0
    p_start, p_end = progress_range
    p_span = max(0, p_end - p_start)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_cut_one, scene): scene for scene in scenes}
        for future in as_completed(futures):
            completed += 1
            if emit_progress_updates:
                pct = p_start + int((completed / total) * p_span)
                emit_progress(pct, f"Cutting scene {completed}/{total}...")
            result = future.result()
            results.append(result)
            if on_ready is not None:
                on_ready(result)

    return results
