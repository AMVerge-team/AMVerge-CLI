from __future__ import annotations

"""Smart scene cutting: lossless copy backed by the container's own edit
list, re-encode as the fallback.

ffmpeg's own mp4/mov muxer already does almost everything this module needs
whenever it stream-copies an off-keyframe range: given ``-ss start -i src -c
copy``, it backward-snaps to the real preceding keyframe (the only place a
stream copy can start), flags everything before the true start as
decode-only (`discard`), and writes an edit list (`elst`) that tells any
compliant reader to hide that pre-roll and begin display exactly at
``start``. Verified directly against real playback in After Effects, and
identically for H.264 and HEVC, mp4 and mov, with and without audio -- this
is a general ISOBMFF/mov-muxer behavior, not a codec-specific trick.

What it does *not* do is the same thing for the tail: asking for ``-t
duration`` past a keyframe stream-copies a little further than requested (as
far as decode dependencies require, typically a couple of frames -- never
all the way to the next real keyframe, however far that is) and simply
reports the wrong, overshot duration. `editlist.patch_trailing_duration`
closes that gap by rewriting the edit list's own duration field in place, no
different in kind from what ffmpeg already wrote for the head.

Together these mean a scene can be cut losslessly -- no re-encoded frames,
no concatenation, none of the decode/display-order splice hazards that come
with re-encoding a head and stitching it to a copied tail -- for any
off-keyframe boundary, provided the real preceding keyframe isn't so far
back that copying (and hiding) all the frames in between stops being worth
it. That is a size/seek-latency trade-off (`MAX_PRE_ROLL`), not a
correctness one: past it, re-encoding is simply cheaper, not more correct.
"""

import os
import subprocess
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

from ..infra.binaries import get_ffmpeg, get_ffprobe
from ..infra.ipc import emit_progress, log
from .editlist import patch_trailing_duration

PRE_SEEK_OFFSET = 10.0

# How far back the real preceding keyframe may sit before a lossless copy
# stops being worth it. Past this, a compliant player has to decode that
# many seconds of hidden, never-displayed pre-roll before it can show or
# seek to the scene's first visible frame, and the file carries those bytes
# for no visible benefit. Re-encoding avoids both at a cost that no longer
# looks large by comparison. Tunable; not a correctness boundary.
MAX_PRE_ROLL = 5.0

# Scenes shorter than this always re-encode. Nothing below is incorrect --
# the discard/edit-list mechanism doesn't care how short the visible slice
# is -- but at a few frames long, re-encoding is instant, and it avoids a
# clip that's almost entirely hidden pre-roll for a sliver of real content.
MIN_COPY_DURATION = 0.5


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


def _lossless_copy(input_file: Path, start: float, end: float, out_path: Path) -> None:
    """Stream-copy ``[start, end)``. See module docstring for why this is
    trusted to be frame-accurate without any further re-encoding or trimming
    of the video/audio bitstreams themselves."""
    _run_ffmpeg([
        get_ffmpeg(), "-y",
        "-ss", f"{start:.3f}",
        "-i", str(input_file),
        "-t", f"{end - start:.3f}",
        "-map", "0:v:0", "-map", "0:a?",
        "-c:v", "copy", "-c:a", "copy",
        "-movflags", "+faststart",
        str(out_path),
    ])


def _is_10bit(path: Path) -> bool:
    """Whether the video stream's pixel format carries more than 8 bits per
    channel (ProRes, HEVC Main10, most lossless/intermediate codecs)."""
    try:
        out = subprocess.run(
            [
                get_ffprobe(), "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=pix_fmt", "-of", "default=nk=1:nw=1", str(path),
            ],
            capture_output=True, text=True, **_background_kwargs(),
        ).stdout
    except Exception:
        return False
    pix_fmt = (out or "").strip().lower()
    return pix_fmt.endswith(("10le", "10be", "12le", "12be", "14le", "14be", "16le", "16be"))


def _preceding_keyframe(keyframes: list[float], start_sec: float) -> float | None:
    """The real keyframe a backward-snapping ``-ss`` will land on.

    None only if ``start_sec`` is before every known keyframe -- shouldn't
    happen in practice, since frame 0 of any file is always a keyframe, but
    an empty or malformed ``keyframes`` list is handled the same as a
    genuine miss: fall through to re-encode rather than guess.
    """
    i = bisect_right(keyframes, start_sec)
    return keyframes[i - 1] if i > 0 else None


def _encode_segment(
    input_file: Path, start: float, end: float, out_path: Path, use_cuda: bool
) -> None:
    pre_seek = max(0.0, start - PRE_SEEK_OFFSET)
    post_seek = start - pre_seek
    duration = end - start
    ten_bit = _is_10bit(input_file)

    def _build_cmd(gpu: bool) -> list[str]:
        if gpu:
            enc = ["-c:v", "h264_nvenc", "-preset", "p1", "-rc", "vbr", "-cq", "16", "-b:v", "0"]
        elif ten_bit:
            enc = ["-c:v", "libx264", "-profile:v", "high10", "-preset", "ultrafast", "-crf", "16"]
        else:
            enc = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "16"]
        pix_fmt = "yuv420p10le" if (ten_bit and not gpu) else "yuv420p"
        c = [get_ffmpeg(), "-y"]
        if pre_seek > 0.0:
            c += ["-ss", f"{pre_seek:.3f}"]
        c += ["-i", str(input_file)]
        c += ["-ss", f"{post_seek:.3f}", "-t", f"{duration:.3f}"]
        c += ["-map", "0:v:0", "-map", "0:a?", "-pix_fmt", pix_fmt]
        c += enc
        c += ["-c:a", "aac", "-b:a", "128k", str(out_path)]
        return c

    if use_cuda and not ten_bit:
        try:
            _run_ffmpeg(_build_cmd(gpu=True))
            return
        except Exception as exc:
            log(f"GPU encode failed, falling back to CPU: {exc}")
    _run_ffmpeg(_build_cmd(gpu=False))


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
    - ``copy`` when a preceding keyframe exists within ``MAX_PRE_ROLL``
      seconds of ``start_sec``: a single lossless stream copy, frame-accurate
      on both ends via the container's own edit list (see module docstring).
    - ``reencode`` otherwise -- no keyframe close enough behind ``start_sec``
      to make a lossless copy worthwhile, the scene is shorter than
      ``MIN_COPY_DURATION``, or (rare) the copy's own edit list didn't come
      out in a shape ``editlist.patch_trailing_duration`` can patch.

    Args:
        input_file: Source video file.
        start_sec: Scene start time in seconds.
        end_sec: Scene end time in seconds.
        scene_idx: Scene number, used for output filename ``scene_{idx:04d}.mp4``.
        out_dir: Directory for output clips.
        keyframes: Sorted list of keyframe timestamps.
        use_cuda: If True, use NVENC for re-encode (GPU). CPU fallback if
            encoder not available.
        is_hevc: Accepted for API compatibility with existing callers;
            no longer changes behavior -- the edit-list mechanism this module
            relies on works identically for HEVC and H.264 (verified), so
            there's no separate HEVC path to select any more.

    Returns:
        Tuple of ``(clip_path, mode)`` where ``mode`` is ``"copy"`` or
        ``"reencode"``.

    Raises:
        ValueError: If ``start_sec >= end_sec`` (non-positive duration).
    """
    out_path = out_dir / f"scene_{scene_idx:04d}.mp4"
    duration = end_sec - start_sec

    if duration <= 0:
        raise ValueError(f"Non-positive duration for scene {scene_idx}: {duration:.3f}s")

    if duration >= MIN_COPY_DURATION:
        k_prev = _preceding_keyframe(keyframes, start_sec)
        if k_prev is not None and (start_sec - k_prev) <= MAX_PRE_ROLL:
            try:
                _lossless_copy(input_file, start_sec, end_sec, out_path)
                if not patch_trailing_duration(out_path, duration):
                    raise RuntimeError("copy's edit list wasn't in a patchable shape")
                return str(out_path), "copy"
            except Exception as exc:
                log(f"Scene {scene_idx}: fast copy failed ({exc}), re-encoding instead")

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
        is_hevc: Accepted for API compatibility; see :func:`cut_scene`.
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
