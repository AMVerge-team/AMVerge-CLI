from __future__ import annotations

"""Smart scene cutting: two modes, chosen once per job, not per scene.

- ``copy``: a true, unmodified stream copy. AI-detected scene boundaries
  rarely land on a keyframe (H.264/HEVC can only start decoding cleanly on
  one), so a copy that started exactly at the detected cut would either be
  undecodable or need something else. The clip can carry up to
  one keyframe interval of the neighboring scene at either edge.
- ``reencode``: exact boundaries, at the cost of a real (if usually small)
  re-encode.

Which mode to use is the caller's choice, made once for the whole job --
this module does not guess per scene.
"""

import io
import os
import subprocess
from bisect import bisect_left, bisect_right
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path
from typing import Callable, Literal

import av

from ..export.params import frame_cut_args, frame_grid_range
from ..infra.binaries import get_ffmpeg, get_ffprobe
from ..video.probe_utils import probe_seek_origin, probe_video_rate
from ..infra.ipc import emit_progress, log

CutMode = Literal["copy", "reencode"]

PRE_SEEK_OFFSET = 10.0


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


def snap_range_to_keyframes(
    keyframes: list[float], start_sec: float, end_sec: float, tolerance: float = 1e-6,
    rate: Fraction | None = None,
) -> tuple[float, float | None]:
    """Widen ``[start_sec, end_sec)`` outward to its enclosing keyframes.

    ``clip_start`` is the last keyframe at or before ``start_sec``;
    ``clip_end`` is the first keyframe at or after ``end_sec``, or ``None``
    if ``end_sec`` is past every known keyframe (the scene runs to the end
    of the file -- the caller should copy through EOF rather than pass an
    explicit ``-t``).

    An already-aligned boundary comes back unchanged, since the enclosing
    keyframe in that case *is* the boundary itself -- callers don't need to
    special-case "already on a keyframe" separately from "snap to the
    nearest one".

    A keyframe within ``tolerance`` seconds of a boundary counts as that
    boundary, so float noise between a frame time computed as
    ``frame / fps`` and one read as ``pts * time_base`` cannot push the
    snap out to the next keyframe. Pass the source's ``rate`` to widen that
    to half a frame: containers that store pts on a coarse timescale (1/1000
    is common after an MKV remux) keep a 24000/1001 keyframe up to 0.5 ms off
    its ideal frame time, and a 1 us tolerance then snapped a scene that
    starts exactly on a keyframe back to the previous one. No two frames are
    closer than one frame apart, so a keyframe within half a frame of a
    boundary can only be the boundary frame itself.
    """
    if rate:
        tolerance = max(tolerance, float(Fraction(1, 2) / rate))
    if not keyframes:
        raise ValueError("Cannot stream-copy a scene without keyframe timestamps")

    i = bisect_right(keyframes, start_sec + tolerance)
    if i == 0:
        raise ValueError(
            f"No keyframe at or before scene start {start_sec:.9f}s; cannot snap outward"
        )
    clip_start = keyframes[i - 1]

    j = bisect_left(keyframes, end_sec - tolerance)
    clip_end = keyframes[j] if j < len(keyframes) else None

    return clip_start, clip_end


_TS_EPSILON = 1e-6


def _add_copy_stream(dst, stream):
    """Add an output stream that remuxes ``stream``'s packets untouched.
    ``opaque`` copies the codec parameters without looking up an encoder
    (there is none for a decoder-only codec such as libdav1d AV1). The source
    codec tag is kept, else the MOV muxer labels every ProRes stream HQ
    (``apch``), and HEVC is tagged ``hvc1`` since QuickTime rejects MP4's
    default ``hev1`` (MKV sources carry no tag of their own)."""
    out = dst.add_stream_from_template(stream, opaque=True)
    tag = stream.codec_context.codec_tag
    if stream.type == "video" and stream.codec_context.name == "hevc":
        out.codec_context.codec_tag = "hvc1"
    elif tag.strip("\x00 "):
        out.codec_context.codec_tag = tag
    return out


def copy_muxability_error(input_file: Path, container: str, audio: bool = True) -> dict[str, str] | None:
    """Return details when a target muxer cannot accept copied input streams."""
    with av.open(str(input_file)) as src:
        video_streams = list(src.streams.video)
        if not video_streams:
            return {"stream_type": "video", "stream_index": "-1", "codec": "none", "reason": "no video stream"}
        streams = [*video_streams, *(src.streams.audio if audio else [])]
        for stream in streams:
            try:
                with av.open(io.BytesIO(), "w", format=container) as probe:
                    _add_copy_stream(probe, stream)
                    probe.start_encoding()
            except (ValueError, av.FFmpegError) as e:
                return {
                    "stream_type": stream.type,
                    "stream_index": str(stream.index),
                    "codec": stream.codec_context.name or "unknown",
                    "reason": str(e),
                }
        try:
            with av.open(io.BytesIO(), "w", format=container) as probe:
                for stream in streams:
                    _add_copy_stream(probe, stream)
                probe.start_encoding()
        except (ValueError, av.FFmpegError) as e:
            return {
                "stream_type": "muxer",
                "stream_index": "-1",
                "codec": "multiple",
                "reason": str(e),
            }
    return None


def copy_muxable(input_file: Path, container: str, audio: bool = True) -> bool:
    """Whether ``container`` can hold copied video and optional audio streams."""
    return copy_muxability_error(input_file, container, audio) is None


def _own_container(input_file: Path) -> str | None:
    with av.open(str(input_file)) as src:
        names = src.format.name.split(",")
    if "avi" in names:
        return "avi"
    if "matroska" in names:
        return "mkv"
    return None


def copy_container_suffix(input_file: Path) -> str:
    """File suffix a copy-mode clip of ``input_file`` must use: ``".mp4"``
    when every stream :func:`_lossless_copy` keeps can be muxed into MP4
    as-is, else the source's own AVI or Matroska container, which always
    holds its own streams (HuffYUV, Ut Video, MagicYUV and Lagarith fit
    neither MP4 nor MOV), else ``".mov"`` (ProRes, PCM audio). See
    :func:`copy_muxable`. Raises ValueError when none can.
    """
    own = _own_container(input_file)
    candidates = ["mp4", own, "mov"] if own else ["mp4", "mov"]
    for container in candidates:
        if copy_muxable(input_file, "matroska" if container == "mkv" else container):
            return f".{container}"
    raise ValueError(f"No container can stream-copy {Path(input_file).name}")


def _lossless_copy(
    input_file: Path, start: float, end: float | None, out_path: Path, frame_timed: bool = False,
) -> None:
    """Stream-copy the keyframe-to-keyframe span ``[start, end)`` verbatim by
    remuxing packets with PyAV. Callers pass keyframe-aligned boundaries
    (see :func:`snap_range_to_keyframes`); ``end=None`` copies through EOF.

    The span is cut in decode (file) order, not by a duration limit. A plain
    ``ffmpeg -t`` stream copy compares each packet's *dts* against the limit,
    and on B-frame sources the end keyframe and the reference frames right
    after it still have a dts under the limit -- so the clip used to end with
    a few frames of the next scene, with the B-frames between them missing.
    On a closed GOP everything stored before the end keyframe displays before
    it and everything after displays after it, so stopping at that packet is
    exact.

    Open GOP (x265's default, common in anime encodes) stores *leading*
    pictures right after the keyframe that display before it but can only be
    decoded from it. Dropping them would lose the last frames of the scene,
    so when they are present the end keyframe and its leading pictures are
    kept: the clip then ends with exactly one frame of the next scene (the
    keyframe itself), consistent with this mode's documented edge bleed. At
    the start, leading pictures of the start keyframe are undecodable without
    the previous GOP and display before ``start``, so they are dropped.

    Stream metadata (language tags, handler names) and dispositions are
    carried over, as ``ffmpeg -c copy`` did. Video pts/dts and audio are rebased so the start keyframe is at 0, the
    same timeline an ``ffmpeg -ss X -c copy`` produced. Audio packets are
    kept when they overlap ``[start, end)``, so the packet straddling the
    start (and the AAC priming packet at the head of a file) survive with a
    slightly negative pts, as they did under ffmpeg. Except in AVI: its
    video time base is one frame, so the muxer's shift past that negative
    audio rounds the second frame a whole slot late and the first frame
    shows twice. There audio starts with the first packet at or after the
    keyframe instead, and ends with the last packet that finishes by ``end``
    (a concat spaces clips by their longest stream, and audio running past
    the video leaves an empty frame at the join). AVI also has no pts, only
    a slot per packet in decode order, so the leading pictures dropped at
    the start would leave empty slots after the keyframe; the later video
    dts close that gap.
    """
    frame_timed = frame_timed or out_path.suffix.lower() == ".avi"
    try:
        with av.open(str(input_file)) as src, av.open(
            str(out_path), "w",
            container_options={"movflags": "+faststart"} if out_path.suffix.lower() in (".mp4", ".mov") else {},
        ) as dst:
            video = src.streams.video[0]
            audio = list(src.streams.audio)
            out_streams = {}
            for stream in (video, *audio):
                out = _add_copy_stream(dst, stream)
                out.metadata.update(stream.metadata)
                out.disposition = stream.disposition
                out_streams[stream.index] = out
            dst.metadata.update({
                k: v for k, v in src.metadata.items()
                if k not in ("major_brand", "minor_version", "compatible_brands")
            })

            def seconds(ts: int | None, tb) -> float | None:
                return None if ts is None else float(ts * tb)

            src.seek(int(start / video.time_base), stream=video, backward=True, any_frame=False)

            base: dict[int, int] = {}
            base_sec = None
            started = False
            video_done = False
            held: list = []
            pending_audio: list = []
            audio_done = {s.index: False for s in audio}

            dropped_ticks = 0

            def write(packet) -> None:
                offset = base[packet.stream.index]
                if packet.pts is not None:
                    packet.pts -= offset
                if packet.dts is not None:
                    packet.dts -= offset
                    if packet.stream.index == video.index:
                        packet.dts -= dropped_ticks
                packet.stream = out_streams[packet.stream.index]
                dst.mux(packet)

            def write_audio(packet) -> bool:
                t = seconds(packet.pts if packet.pts is not None else packet.dts, packet.time_base)
                if end is not None and t >= end - _TS_EPSILON:
                    audio_done[packet.stream.index] = True
                    return False
                t_end = t + seconds(packet.duration or 0, packet.time_base)
                if frame_timed and end is not None and t_end > end + _TS_EPSILON:
                    audio_done[packet.stream.index] = True
                    return False
                first = t if frame_timed else t_end
                if first >= float(base_sec) - _TS_EPSILON:
                    write(packet)
                return True

            def set_base(key_packet) -> None:
                nonlocal base_sec
                base_sec = key_packet.pts * video.time_base
                base[video.index] = key_packet.pts
                for s in audio:
                    base[s.index] = round(base_sec / s.time_base)

            rate = video.average_rate or video.guessed_rate
            frame_ticks = round(1 / (rate * video.time_base)) if rate else 1

            for packet in src.demux(video, *audio):
                if packet.dts is None and packet.pts is None:
                    continue

                if packet.stream.index == video.index:
                    if video_done:
                        if all(audio_done.values()):
                            break
                        continue
                    t = seconds(packet.pts, video.time_base)
                    if not started:
                        if packet.is_keyframe and t is not None and t >= start - _TS_EPSILON:
                            started = True
                            set_base(packet)
                            write(packet)
                            for p in pending_audio:
                                write_audio(p)
                            pending_audio = []
                        continue
                    if held:
                        if t is not None and t < end - _TS_EPSILON:
                            held.append(packet)
                            continue
                        if len(held) > 1:
                            for p in held:
                                write(p)
                        held = []
                        video_done = True
                        continue
                    if end is not None and packet.is_keyframe and t is not None and t >= end - _TS_EPSILON:
                        held = [packet]
                        continue
                    if t is not None and t < start - _TS_EPSILON:
                        if frame_timed:
                            dropped_ticks += packet.duration or frame_ticks
                        continue
                    write(packet)
                    continue

                if not started:
                    pending_audio.append(packet)
                    continue
                if not write_audio(packet) and video_done and all(audio_done.values()):
                    break

            if len(held) > 1:
                for p in held:
                    write(p)

            if not started:
                raise RuntimeError(f"No keyframe at or after {start:.6f}s in {input_file}")
    except Exception:
        try:
            out_path.unlink()
        except OSError:
            pass
        raise


def copy_range(
    input_file: Path,
    start_sec: float,
    end_sec: float,
    out_path: Path,
    keyframes: list[float],
    rate: Fraction | None = None,
    frame_timed: bool = False,
) -> float:
    """Stream-copy ``[start_sec, end_sec)`` widened outward to its enclosing
    keyframes. The only copy-mode cut: :func:`cut_scene` and
    ``export.engine._smartcut_ranges`` both go through it, so a boundary
    fix lands in both at once. ``frame_timed`` starts audio at the keyframe
    as for an AVI output (see :func:`_lossless_copy`); set it when the clip
    is headed for AVI in a later step.

    The range is first mapped onto the source's frame grid (recovering the
    intended frame from millisecond-rounded export times), then snapped to
    keyframes with a half-frame tolerance, then remuxed by
    :func:`_lossless_copy`. ``rate`` is probed when not given; pass it when
    cutting many ranges from one source.

    Returns how far into the clip the requested start sits, in whole frames
    when the rate is known -- 0.0 when the range already started on a
    keyframe.
    """
    if rate is None:
        rate = probe_video_rate(input_file)
    frames = frame_grid_range(start_sec, end_sec, rate)
    if frames:
        start_sec, end_sec = float(frames[0] / rate), float(frames[1] / rate)
    clip_start, clip_end = snap_range_to_keyframes(keyframes, start_sec, end_sec, rate=rate)
    _lossless_copy(Path(input_file), clip_start, clip_end, Path(out_path), frame_timed)
    offset = start_sec - clip_start
    if rate:
        offset = float(round(offset * rate) / rate)
    return offset


_PREVIEW_AUDIO_CODECS = {"aac", "mp3", "opus", "flac"}


def exact_copy_supported(input_file: Path) -> bool:
    """Whether scenes of ``input_file`` can be stream-copied into previews the
    app plays as-is: the copy lands in MP4 (see :func:`copy_container_suffix`)
    and every audio stream is one Chromium plays from MP4 (AAC, MP3, Opus,
    FLAC). AC3/DTS/TrueHD and anything else keep the whole job on re-encode."""
    try:
        if copy_container_suffix(input_file) != ".mp4":
            return False
        with av.open(str(input_file)) as src:
            return all(s.codec_context.name in _PREVIEW_AUDIO_CODECS for s in src.streams.audio)
    except (ValueError, OSError, av.FFmpegError):
        return False


def split_exact_copy_scenes(
    scenes: list[dict],
    keyframes: list[float],
    open_gop: list[float],
    rate: Fraction | None,
    duration: float | None,
) -> tuple[list[dict], list[dict]]:
    """Split contiguous scenes into ``(copy, reencode)`` batches, where a copy
    is exactly as frame-accurate as a re-encode.

    A scene goes to ``copy`` only when :func:`copy_range` would cut it with no
    bleed at all: it sits on the constant frame grid, its start is a keyframe,
    and its end is a keyframe without open-GOP leading pictures (those keep
    one frame of the next scene, see :func:`_lossless_copy`) -- or, for the
    last scene, the end of the file. Both checks use the same half-frame
    snap :func:`copy_range` does. Everything else goes to ``reencode``.
    """
    if not rate or not keyframes:
        return [], list(scenes)

    half = float(Fraction(1, 2) / rate)
    copy: list[dict] = []
    reencode: list[dict] = []
    for pos, scene in enumerate(scenes):
        frames = frame_grid_range(float(scene["start_sec"]), float(scene["end_sec"]), rate)
        if frames is None:
            reencode.append(scene)
            continue
        start, end = float(frames[0] / rate), float(frames[1] / rate)
        try:
            clip_start, clip_end = snap_range_to_keyframes(keyframes, start, end, rate=rate)
        except ValueError:
            reencode.append(scene)
            continue

        start_exact = abs(clip_start - start) <= half
        if clip_end is None:
            # copies through EOF, exact only for a last scene that ends there
            end_exact = pos == len(scenes) - 1 and duration is not None and abs(duration - end) <= half
        else:
            end_exact = abs(clip_end - end) <= half and not any(abs(clip_end - k) <= half for k in open_gop)

        (copy if start_exact and end_exact else reencode).append(scene)
    return copy, reencode


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


def _encode_segment(
    input_file: Path, start: float, end: float, out_path: Path, use_cuda: bool
) -> None:
    """Re-encode exactly the frames in ``[start, end)``. On a constant frame
    grid the cut is a frame selection (see
    :func:`~amverge.core.export.params.frame_cut_args`), immune to the float
    rounding of ``start``/``end``; off-grid (VFR) it falls back to plain
    time-based seeking."""
    rate = probe_video_rate(input_file)
    origin = probe_seek_origin(input_file)
    frames = frame_grid_range(start, end, rate)
    if frames:
        cut = frame_cut_args(*frames, rate, origin)
        in_args = cut.input_args
        out_args = [*cut.output_args, "-r", f"{rate.numerator}/{rate.denominator}", "-fps_mode", "cfr"]
        audio_chain = cut.audio_filters("asetpts=PTS-STARTPTS")
    else:
        pre_seek = max(0.0, start - PRE_SEEK_OFFSET)
        in_args = ["-ss", f"{pre_seek + float(origin):.9f}"] if pre_seek > 0.0 else []
        out_args = ["-ss", f"{start - pre_seek:.9f}", "-t", f"{end - start:.9f}"]
        audio_chain = "asetpts=PTS-STARTPTS"
    ten_bit = _is_10bit(input_file)

    def _build_cmd(gpu: bool) -> list[str]:
        if gpu:
            enc = ["-c:v", "h264_nvenc", "-preset", "p1", "-rc", "vbr", "-cq", "16", "-b:v", "0"]
        elif ten_bit:
            enc = ["-c:v", "libx264", "-profile:v", "high10", "-preset", "ultrafast", "-crf", "16"]
        else:
            enc = ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "16"]
        pix_fmt = "yuv420p10le" if (ten_bit and not gpu) else "yuv420p"
        c = [get_ffmpeg(), "-y", *in_args, "-i", str(input_file), *out_args]
        c += ["-map", "0:v:0", "-map", "0:a?", "-pix_fmt", pix_fmt]
        c += ["-vf", "setpts=PTS-STARTPTS", "-af", audio_chain]
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
    mode: CutMode,
    use_cuda: bool = False,
) -> tuple[str, str, float]:
    """Cut a single scene using the job's chosen mode.

    Args:
        input_file: Source video file.
        start_sec: Scene start time in seconds (the AI-detected cut).
        end_sec: Scene end time in seconds.
        scene_idx: Scene number, used for output filename ``scene_{idx:04d}.mp4``
            (``.mov`` for a copy-mode clip whose source MP4 cannot hold, see
            :func:`copy_container_suffix`).
        out_dir: Directory for output clips.
        keyframes: Sorted list of keyframe timestamps.
        mode: ``"copy"`` widens the range outward to the enclosing keyframes
            and does one plain stream copy (see module docstring for why
            and what it costs); ``"reencode"`` re-encodes the exact
            ``[start_sec, end_sec)`` range with no widening.
        use_cuda: If True and ``mode == "reencode"``, use NVENC (GPU), with
            CPU fallback if the encoder isn't available. Unused in copy mode.

    Returns:
        ``(clip_path, mode, poster_offset_sec)``. ``poster_offset_sec`` is
        how far into the clip the scene's true first frame actually sits --
        always 0.0 for ``reencode`` (exact boundary, nothing to skip), and
        ``start_sec - clip_start`` for ``copy`` (also 0.0 when the scene
        already started on a keyframe). A caller generating a poster
        thumbnail should seek this far into the clip first, or it will grab
        a frame of bleed from the previous scene instead.

    Raises:
        ValueError: If ``start_sec >= end_sec``, or ``mode`` isn't recognized.
        RuntimeError: If the underlying ffmpeg invocation fails. Copy mode
            never silently falls back to re-encoding -- a mode of ``"copy"``
            means every frame of the output is a real stream copy, or the
            scene fails outright and the caller finds out.
    """
    duration = end_sec - start_sec
    if duration <= 0:
        raise ValueError(f"Non-positive duration for scene {scene_idx}: {duration:.3f}s")

    out_path = out_dir / f"scene_{scene_idx:04d}.mp4"

    if mode == "copy":
        out_path = out_path.with_suffix(copy_container_suffix(input_file))
        offset = copy_range(input_file, start_sec, end_sec, out_path, keyframes)
        return str(out_path), "copy", offset
    elif mode == "reencode":
        _encode_segment(input_file, start_sec, end_sec, out_path, use_cuda)
        return str(out_path), "reencode", 0.0
    else:
        raise ValueError(f"Unknown cut mode: {mode!r} (expected 'copy' or 'reencode')")


def cut_all_scenes(
    input_file: Path,
    scenes: list[dict],
    keyframes: list[float],
    out_dir: Path,
    mode: CutMode,
    use_cuda: bool = False,
    max_workers: int = 4,
    on_ready: Callable[[dict], None] | None = None,
    progress_range: tuple[int, int] = (82, 97),
    emit_progress_updates: bool = True,
) -> list[dict]:
    """Cut multiple scenes in parallel using a thread pool.

    Each scene dict must have ``"scene_index"``, ``"start_sec"``,
    and ``"end_sec"`` keys. Cutting happens via :func:`cut_scene`, in the
    same ``mode`` for every scene in the batch -- see its docstring for what
    each mode means and guarantees.

    Args:
        input_file: Source video file.
        scenes: List of scene dicts with ``scene_index``, ``start_sec``,
            ``end_sec``.
        keyframes: Sorted keyframe timestamps from
            :func:`~amverge.core.keyframe_align.get_keyframe_timestamps_pyav`.
        out_dir: Directory for output clips.
        mode: ``"copy"`` or ``"reencode"``, applied uniformly to every scene
            in this batch.
        use_cuda: Enable NVENC GPU encode when ``mode == "reencode"``.
        max_workers: Thread pool size.
        on_ready: Called per completed scene with ``{"scene_index": int,
            "clip_path": str | None, "clip_mode": str, "poster_offset_sec":
            float}``.
        progress_range: ``(start, end)`` tuple for IPC progress percentage
            during cutting.
        emit_progress_updates: If True, emit ``PROGRESS|pct|msg`` IPC events.

    Returns:
        List of result dicts, same shape as passed to ``on_ready``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(scenes)
    results: list[dict] = []
    if total == 0:
        return results

    def _cut_one(scene: dict) -> dict:
        idx = scene["scene_index"]
        try:
            clip_path, clip_mode, poster_offset = cut_scene(
                input_file,
                float(scene["start_sec"]),
                float(scene["end_sec"]),
                idx,
                out_dir,
                keyframes,
                mode,
                use_cuda,
            )
            log(f"Scene {idx}: {clip_mode} -> {Path(clip_path).name}")
            return {
                "scene_index": idx, "clip_path": clip_path,
                "clip_mode": clip_mode, "poster_offset_sec": poster_offset,
            }
        except Exception as exc:
            log(f"Warning: scene {idx} failed: {exc}")
            return {
                "scene_index": idx, "clip_path": None,
                "clip_mode": "failed", "poster_offset_sec": 0.0,
            }

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
