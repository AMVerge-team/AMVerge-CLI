"""Export orchestration: per-clip copy/re-encode with fallbacks, merge, parallel."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from ..infra.binaries import get_ffmpeg, get_ffprobe
from . import params

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

ProgressCb = Callable[[int, str], None]  # (percent 0-100, message)
EventCb = Callable[[str], None]          # raw event line, e.g. "CLIP_READY|3|<path>|copy"


class ExportError(RuntimeError):
    pass


class ExportAborted(RuntimeError):
    pass


@dataclass
class ExportSettings:
    codec: str = "copy"       # "copy" | h264_* | h265_* | av1_main | prores_*
    audio: str = "copy"
    container: str = "mp4"
    hardware: str = "auto"    # auto | gpu | cpu
    merge: bool = False
    workers: int = 1
    audio_track: Optional[int] = None  # 0-based audio index to hoist to first (preview language)
    audio_language: Optional[str] = None  # language tag to hoist; resolved per input
    audio_single: bool = False  # keep only the selected track (mixed-layout merges)


@dataclass
class ExportJob:
    """One scene to export. ``seek_ms``/``dur_ms`` set → cut that range from a
    source episode; both None → ``input`` is a whole pre-cut clip file."""
    scene_index: int
    input: str
    seek_ms: Optional[int] = None
    dur_ms: Optional[int] = None


# -- probing / gpu ---------------------------------------------------------

def detect_gpu_encoders(ffmpeg: str) -> set[str]:
    try:
        r = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=20, creationflags=CREATE_NO_WINDOW,
        )
    except Exception:
        return set()
    out = r.stdout or ""
    return {enc for enc in ("h264_nvenc", "hevc_nvenc", "av1_nvenc") if enc in out}


def _cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_use_gpu(settings: ExportSettings, gpu_encoders: set[str]) -> bool:
    """GPU only when requested, the codec has an NVENC path, and that encoder is
    actually present in this ffmpeg build."""
    if settings.codec == "copy" or params.codec_family(settings.codec) == "prores":
        return False
    if settings.hardware == "cpu":
        return False
    want = settings.hardware == "gpu" or (settings.hardware == "auto" and _cuda_available())
    if not want:
        return False
    entry = params.VIDEO_PARAMS.get(params.normalize_codec(settings.codec))
    if not entry or entry["gpu"] is None:
        return False
    return entry["gpu"][0] in gpu_encoders  # type: ignore[index]


def probe_video_codec(ffprobe: str, path: str) -> Optional[str]:
    return _probe_stream(ffprobe, path, "v")


def probe_audio_codec(ffprobe: str, path: str) -> Optional[str]:
    return _probe_stream(ffprobe, path, "a")


def _probe_stream(ffprobe: str, path: str, kind: str) -> Optional[str]:
    try:
        r = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", f"{kind}:0",
             "-show_entries", "stream=codec_name", "-of", "default=nk=1:nw=1", path],
            capture_output=True, text=True, timeout=20, creationflags=CREATE_NO_WINDOW,
        )
        name = (r.stdout or "").strip().splitlines()
        return name[0].strip().lower() if name and name[0].strip() else None
    except Exception:
        return None


def _all_same_video_codec(ffprobe: str, inputs: list[str]) -> bool:
    """True if every input shares one (known) video codec — required for a clean
    direct concat stream-copy."""
    seen: set = set()
    for inp in inputs:
        seen.add(probe_video_codec(ffprobe, inp))
        if len(seen) > 1:
            return False
    return len(seen) == 1 and None not in seen


def _copy_concat_safe(ffprobe: str, path: str) -> bool:
    """True if `path` can be stream-copied into a concat without artifacts.

    A clip whose first displayed frame is not a keyframe, or whose video stream
    starts after 0, joins badly: the decoder holds the last frame of the previous
    clip while it waits for a keyframe, and audio keeps running underneath. That
    is the freeze-then-echo at a join. Such clips go down the re-encode path
    instead, which rebases timestamps and lands on a clean keyframe.
    """
    try:
        r = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "frame=key_frame,pts_time",
             "-read_intervals", "%+#1", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=20, creationflags=CREATE_NO_WINDOW,
        )
        first = next((ln for ln in (r.stdout or "").splitlines() if ln.strip()), "")
        if not first:
            return False
        parts = [p.strip() for p in first.split(",")]
        if parts[0] != "1":
            return False
        start = float(parts[1]) if len(parts) > 1 and parts[1] else 0.0
        return start < 0.001
    except Exception:
        return False


def probe_audio_stream_count(ffprobe: str, path: str) -> int:
    try:
        r = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=20, creationflags=CREATE_NO_WINDOW,
        )
        return len([ln for ln in (r.stdout or "").splitlines() if ln.strip()])
    except Exception:
        return 0


def probe_audio_languages(ffprobe: str, path: str) -> list[str]:
    """Language tag of each audio stream, in order. Untagged streams give ""."""
    try:
        r = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "a",
             "-show_entries", "stream_tags=language", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=20, creationflags=CREATE_NO_WINDOW,
        )
        return [ln.strip().lower() for ln in (r.stdout or "").splitlines()]
    except Exception:
        return []


def resolve_audio_track(languages: list[str], language: Optional[str]) -> Optional[int]:
    """Index of the wanted language in this file, or None to leave order alone."""
    if not language or not languages:
        return None
    try:
        return languages.index(language.lower())
    except ValueError:
        return None


def probe_duration_ms(ffprobe: str, path: str) -> Optional[int]:
    try:
        r = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nk=1:nw=1", path],
            capture_output=True, text=True, timeout=20, creationflags=CREATE_NO_WINDOW,
        )
        s = (r.stdout or "").strip()
        return int(float(s) * 1000) if s else None
    except Exception:
        return None


# -- ffmpeg runner ---------------------------------------------------------

def _run_ffmpeg(
    ffmpeg: str,
    args: list[str],
    total_ms: Optional[int],
    on_frac: Optional[Callable[[float], None]],
    abort: threading.Event,
    active: set,
) -> None:
    if abort.is_set():
        raise ExportAborted()

    full = [ffmpeg, "-progress", "pipe:1", "-nostats", "-hide_banner", *args]
    proc = subprocess.Popen(
        full, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, creationflags=CREATE_NO_WINDOW,
    )
    active.add(proc)
    stderr_tail: list[str] = []

    def _drain_err() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            stderr_tail.append(line)
            if len(stderr_tail) > 60:
                del stderr_tail[0]

    err_thread = threading.Thread(target=_drain_err, daemon=True)
    err_thread.start()

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            if abort.is_set():
                proc.kill()
                break
            line = line.strip()
            # ffmpeg -progress emits out_time_us in microseconds.
            if total_ms and on_frac and line.startswith("out_time_us="):
                try:
                    cur_ms = int(line[len("out_time_us="):]) / 1000.0
                    on_frac(max(0.0, min(1.0, cur_ms / total_ms)))
                except ValueError:
                    pass
            elif line == "progress=end":
                break
    finally:
        proc.wait()
        active.discard(proc)
        err_thread.join(timeout=1)

    if abort.is_set():
        raise ExportAborted()
    if proc.returncode != 0:
        raise ExportError("".join(stderr_tail).strip() or f"ffmpeg exited {proc.returncode}")


def _is_gpu_session_error(text: str) -> bool:
    t = text.lower()
    if not any(k in t for k in ("nvenc", "openencodesessionex", "cuda", "_amf", "qsv")):
        return False
    return any(k in t for k in (
        "openencodesessionex failed", "no capable devices", "cannot load",
        "error while opening encoder", "failed to initialise", "failed to initialize",
        "device", "not found", "function not implemented", "invalid argument",
    ))


# -- single-clip export with fallback ladder -------------------------------

def _export_one(
    job: ExportJob,
    out_path: str,
    settings: ExportSettings,
    ffmpeg: str,
    use_gpu: bool,
    source_video_codec: Optional[str],
    total_ms: Optional[int],
    on_frac: Optional[Callable[[float], None]],
    abort: threading.Event,
    active: set,
    audio_count: Optional[int] = None,
    audio_languages: Optional[list[str]] = None,
) -> str:
    """Export one scene to ``out_path``, returning the mode used ("copy" or
    "reencode"). Ladder: copy → re-encode → CPU re-encode."""
    ext = Path(out_path).suffix.lstrip(".").lower()
    track = settings.audio_track
    if audio_languages:
        resolved = resolve_audio_track(audio_languages, settings.audio_language)
        if resolved is not None:
            track = resolved
            audio_count = len(audio_languages)

    def copy_args() -> list[str]:
        bsf = params.stream_copy_bsf(source_video_codec, ext)
        return params.build_copy_args(job.input, out_path, settings.audio, job.seek_ms, job.dur_ms, bsf,
                                      track, audio_count, settings.audio_single)

    def reencode_args(codec: str, gpu: bool) -> list[str]:
        return params.build_reencode_args(job.input, out_path, codec, settings.audio, gpu,
                                          job.seek_ms, job.dur_ms, track, audio_count,
                                          settings.audio_single)

    if settings.codec == "copy":
        try:
            _run_ffmpeg(ffmpeg, copy_args(), total_ms, on_frac, abort, active)
            return "copy"
        except ExportAborted:
            raise
        except ExportError:
            # Stream copy failed (mismatched params / non-copy-safe) → re-encode.
            return _reencode_with_fallback(job, out_path, "h264_high", settings, ffmpeg,
                                           use_gpu, total_ms, on_frac, abort, active, reencode_args)
    else:
        return _reencode_with_fallback(job, out_path, settings.codec, settings, ffmpeg,
                                       use_gpu, total_ms, on_frac, abort, active, reencode_args)


def _reencode_with_fallback(
    job, out_path, codec, settings, ffmpeg, use_gpu, total_ms, on_frac, abort, active, reencode_args,
) -> str:
    try:
        _run_ffmpeg(ffmpeg, reencode_args(codec, use_gpu), total_ms, on_frac, abort, active)
        return "reencode"
    except ExportAborted:
        raise
    except ExportError as e:
        if use_gpu and _is_gpu_session_error(str(e)):
            _run_ffmpeg(ffmpeg, reencode_args(codec, False), total_ms, on_frac, abort, active)
            return "reencode"
        raise


def _smartcut_ranges(jobs: list[ExportJob], tmp_dir: Path, use_cuda: bool) -> list[ExportJob]:
    """For remux (copy) exports, cut each source range with smart_cut — stream-copy
    the keyframe-aligned GOPs and re-encode only the leading/trailing edges — so
    boundaries are frame-accurate and mostly lossless. Whole-file jobs pass through.
    Returns jobs rewritten to point at the cut temp clips (no range)."""
    from ..cutting.smart_cut import cut_scene
    from ..keyframes.keyframe_align import get_keyframe_timestamps_pyav
    from ..codec.codec_utils import check_if_hevc

    kf_cache: dict = {}
    hevc_cache: dict = {}
    out: list[ExportJob] = []
    for job in jobs:
        if job.seek_ms is None:
            out.append(job)
            continue
        src = job.input
        if src not in kf_cache:
            kf_cache[src] = get_keyframe_timestamps_pyav(src)
            hevc_cache[src] = check_if_hevc(src)
        start = job.seek_ms / 1000.0
        end = start + (job.dur_ms or 0) / 1000.0
        clip_path, _mode = cut_scene(
            Path(src), start, end, job.scene_index, tmp_dir, kf_cache[src], use_cuda, hevc_cache[src]
        )
        out.append(ExportJob(scene_index=job.scene_index, input=clip_path))
    return out


# -- orchestration ---------------------------------------------------------

def export_scenes(
    jobs: list[ExportJob],
    out_dir: str,
    file_stem: str,
    settings: ExportSettings,
    *,
    on_progress: Optional[ProgressCb] = None,
    on_event: Optional[EventCb] = None,
    abort: Optional[threading.Event] = None,
    ffmpeg: Optional[str] = None,
    ffprobe: Optional[str] = None,
) -> list[str]:
    """Export ``jobs`` into ``out_dir``. Returns the produced file paths."""
    ffmpeg = ffmpeg or get_ffmpeg()
    ffprobe = ffprobe or get_ffprobe()
    abort = abort or threading.Event()
    active: set = set()
    os.makedirs(out_dir, exist_ok=True)

    def progress(pct: int, msg: str) -> None:
        if on_progress:
            on_progress(max(0, min(100, pct)), msg)

    def event(line: str) -> None:
        if on_event:
            on_event(line)

    gpu_encoders = detect_gpu_encoders(ffmpeg)
    use_gpu = resolve_use_gpu(settings, gpu_encoders)

    # Remux (copy) of source ranges (webp mode) → smartcut each range to an
    # accurate temp clip first, then export those as whole files. Encode
    # workflows already cut frame-accurately via re-encode, so skip them.
    smartcut_tmp = None
    if settings.codec == "copy" and any(j.seek_ms is not None for j in jobs):
        smartcut_tmp = tempfile.TemporaryDirectory()
        use_cuda = settings.hardware != "cpu" and "h264_nvenc" in gpu_encoders
        progress(5, "Cutting source ranges...")
        jobs = _smartcut_ranges(jobs, Path(smartcut_tmp.name), use_cuda)

    source_video_codec = probe_video_codec(ffprobe, jobs[0].input) if jobs else None
    audio_count = (
        probe_audio_stream_count(ffprobe, jobs[0].input)
        if settings.audio_track and jobs else None
    )

    try:
        if settings.merge:
            out = _merge(jobs, out_dir, file_stem, settings, ffmpeg, ffprobe, use_gpu,
                         source_video_codec, progress, event, abort, active, audio_count)
            progress(100, "Export complete")
            return [out]

        return _individual(jobs, out_dir, file_stem, settings, ffmpeg, ffprobe, use_gpu,
                           source_video_codec, progress, event, abort, active, audio_count)
    finally:
        if smartcut_tmp is not None:
            smartcut_tmp.cleanup()


def _out_name(out_dir: str, file_stem: str, scene_index: int, container: str) -> str:
    if "####" in file_stem:
        stem = file_stem.replace("####", f"{scene_index:04d}")
    else:
        stem = f"{file_stem}_{scene_index:04d}"
    return str(Path(out_dir) / f"{stem}.{container}")


def _individual(
    jobs, out_dir, file_stem, settings, ffmpeg, ffprobe, use_gpu,
    source_video_codec, progress, event, abort, active, audio_count=None,
) -> list[str]:
    total = len(jobs)
    workers = max(1, min(settings.workers, total))
    done = 0
    done_lock = threading.Lock()
    outputs: dict[int, str] = {}

    def run_job(job: ExportJob) -> None:
        nonlocal done
        if abort.is_set():
            raise ExportAborted()
        out_path = _out_name(out_dir, file_stem, job.scene_index, settings.container)
        total_ms = job.dur_ms or probe_duration_ms(ffprobe, job.input)
        langs = probe_audio_languages(ffprobe, job.input) if settings.audio_language else None
        mode = _export_one(job, out_path, settings, ffmpeg, use_gpu, source_video_codec,
                           total_ms, None, abort, active, audio_count, langs)
        with done_lock:
            done += 1
            outputs[job.scene_index] = out_path
            progress(int(done / total * 100), f"Exported {done}/{total} clips")
            event(f"CLIP_READY|{job.scene_index}|{out_path}|{mode}")

    if workers <= 1:
        for job in jobs:
            run_job(job)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_job, job) for job in jobs]
            for fut in as_completed(futures):
                fut.result()  # propagate the first error

    return [outputs[idx] for idx in sorted(outputs)]


def _merge(
    jobs, out_dir, file_stem, settings, ffmpeg, ffprobe, use_gpu,
    source_video_codec, progress, event, abort, active, audio_count=None,
) -> str:
    ext = settings.container
    merged_stem = file_stem.replace("_####", "").replace("####", "").strip("_") or "merged"
    out_path = str(Path(out_dir) / f"{merged_stem}.{ext}")

    has_ranges = any(j.seek_ms is not None for j in jobs)
    force_reencode = settings.codec != "copy"  # encode workflow always re-encodes

    if settings.codec == "copy" and not has_ranges:
        inputs = [j.input for j in jobs]
        copy_safe = _all_same_video_codec(ffprobe, inputs) and all(
            _copy_concat_safe(ffprobe, inp) for inp in inputs
        )
        if copy_safe:
            progress(30, "Merging (stream copy)...")
            merge_langs = probe_audio_languages(ffprobe, inputs[0])
            merge_track = resolve_audio_track(merge_langs, settings.audio_language)
            try:
                _concat_copy(inputs, out_path, ext, source_video_codec,
                             settings.audio, ffmpeg, abort, active,
                             merge_track if merge_track is not None else settings.audio_track,
                             len(merge_langs) if merge_langs else audio_count)
                event(f"CLIP_READY|0|{out_path}|copy")
                return out_path
            except ExportAborted:
                raise
            except ExportError:
                progress(35, "Stream-copy merge failed; re-encoding...")
                force_reencode = True
        else:
            force_reencode = True

    seg_copy = settings.codec == "copy" and not force_reencode
    seg_codec = "copy" if seg_copy else (settings.codec if settings.codec != "copy" else "h264_high")
    # Copied audio keeps its original timestamps while the re-encoded video is
    # rebased to 0, so the two drift apart from the first join onward. Encoding
    # the audio lets asetpts/aresample rebase it the same way. Only for segments
    # that feed a concat; a straight copy merge still passes audio through.
    seg_audio = "aac" if (not seg_copy and settings.audio == "copy") else settings.audio

    # concat needs every segment to carry the same streams. Clips that already
    # agree keep all their tracks; a mixed set is narrowed to one track so the
    # join succeeds, and that track is the language asked for, not an arbitrary
    # first one.
    seg_languages = [probe_audio_languages(ffprobe, job.input) for job in jobs]
    uniform_audio = len({tuple(langs) for langs in seg_languages}) == 1
    seg_audio_count = len(seg_languages[0]) if uniform_audio and seg_languages else audio_count
    if not uniform_audio:
        progress(30, "Clips carry different audio tracks; keeping the selected language only")

    seg_settings = ExportSettings(codec=seg_codec, audio=seg_audio, container=ext,
                                  hardware=settings.hardware, merge=False,
                                  audio_track=settings.audio_track,
                                  audio_language=settings.audio_language,
                                  audio_single=not uniform_audio)
    verb = "Remuxing" if seg_copy else "Encoding"
    with tempfile.TemporaryDirectory() as tmp:
        segments: list[str] = []
        total = len(jobs)

        for i, job in enumerate(jobs):
            if abort.is_set():
                raise ExportAborted()
            
            seg = str(Path(tmp) / f"seg_{i:04d}.{ext}")
            total_ms = job.dur_ms or probe_duration_ms(ffprobe, job.input)
            _export_one(job, seg, seg_settings, ffmpeg,
                        resolve_use_gpu(seg_settings, detect_gpu_encoders(ffmpeg)),
                        source_video_codec, total_ms, None, abort, active,
                        seg_audio_count, seg_languages[i] if seg_languages else None)
            segments.append(seg)
            progress(30 + int((i + 1) / total * 55), f"{verb} {i + 1}/{total} segments")

        progress(90, "Concatenating segments...")
        _concat_copy(segments, out_path, ext, source_video_codec if seg_copy else None,
                     "copy", ffmpeg, abort, active)
        
    event(f"CLIP_READY|0|{out_path}|{'copy' if seg_copy else 'reencode'}")
    return out_path


def _concat_copy(
    inputs: list[str], out_path: str, ext: str, source_video_codec: Optional[str],
    audio: str, ffmpeg: str, abort: threading.Event, active: set,
    audio_track: Optional[int] = None, audio_count: Optional[int] = None,
) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as f:
        listfile = f.name
        for inp in inputs:
            safe = inp.replace("'", "'\\''")
            f.write(f"file '{safe}'\n")
    try:
        args = ["-y", "-f", "concat", "-safe", "0", "-i", listfile, "-map", "0:v:0"]
        args += params.audio_map_args(audio, audio_track, audio_count)
        args += ["-c:v", "copy"]
        if audio == "none":
            args += ["-an"]
        else:
            args += ["-c:a", "copy"]
        args += params.disposition_args(audio, audio_track, audio_count)
        bsf = params.stream_copy_bsf(source_video_codec, ext)
        if bsf:
            args += ["-bsf:v", bsf]
        if ext in ("mp4", "mov"):
            args += ["-movflags", "+faststart"]
        args += ["-max_muxing_queue_size", "1024", out_path]
        _run_ffmpeg(ffmpeg, args, None, None, abort, active)
    finally:
        try:
            os.unlink(listfile)
        except OSError:
            pass
