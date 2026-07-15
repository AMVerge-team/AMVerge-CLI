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
) -> str:
    """Export one scene to ``out_path``, returning the mode used ("copy" or
    "reencode"). Ladder: copy → re-encode → CPU re-encode."""
    ext = Path(out_path).suffix.lstrip(".").lower()
    track = settings.audio_track

    def copy_args() -> list[str]:
        bsf = params.stream_copy_bsf(source_video_codec, ext)
        return params.build_copy_args(job.input, out_path, settings.audio, job.seek_ms, job.dur_ms, bsf,
                                      track, audio_count)

    def reencode_args(codec: str, gpu: bool) -> list[str]:
        return params.build_reencode_args(job.input, out_path, codec, settings.audio, gpu,
                                          job.seek_ms, job.dur_ms, track, audio_count)

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
    source_video_codec = probe_video_codec(ffprobe, jobs[0].input) if jobs else None
    audio_count = (
        probe_audio_stream_count(ffprobe, jobs[0].input)
        if settings.audio_track and jobs else None
    )

    if settings.merge:
        out = _merge(jobs, out_dir, file_stem, settings, ffmpeg, ffprobe, use_gpu,
                     source_video_codec, progress, event, abort, active, audio_count)
        progress(100, "Export complete")
        return [out]

    return _individual(jobs, out_dir, file_stem, settings, ffmpeg, ffprobe, use_gpu,
                       source_video_codec, progress, event, abort, active, audio_count)


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
        mode = _export_one(job, out_path, settings, ffmpeg, use_gpu, source_video_codec,
                           total_ms, None, abort, active, audio_count)
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

    whole_file_copy = settings.codec == "copy" and all(j.seek_ms is None for j in jobs)

    if whole_file_copy:
        # Fast path: direct concat stream-copy of the pre-cut clip files.
        progress(30, "Merging (stream copy)...")
        try:
            _concat_copy([j.input for j in jobs], out_path, ext, source_video_codec,
                         settings.audio, ffmpeg, abort, active,
                         settings.audio_track, audio_count)
            event(f"CLIP_READY|0|{out_path}|copy")
            return out_path
        except ExportAborted:
            raise
        except ExportError:
            progress(35, "Stream-copy merge failed; re-encoding...")

    # Segmented path: materialize each job to a temp segment (copy or re-encode),
    # then concat-copy. Handles mixed codecs, source ranges, and encode workflows.
    seg_settings = settings
    if settings.codec == "copy":
        seg_settings = ExportSettings(codec="h264_high", audio=settings.audio,
                                      container=ext, hardware=settings.hardware, merge=False,
                                      audio_track=settings.audio_track)
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
                        source_video_codec, total_ms, None, abort, active, audio_count)
            segments.append(seg)
            progress(30 + int((i + 1) / total * 55), f"Encoded {i + 1}/{total} segments")
        progress(90, "Concatenating segments...")
        _concat_copy(segments, out_path, ext, None, "copy", ffmpeg, abort, active)
    event(f"CLIP_READY|0|{out_path}|reencode")
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
