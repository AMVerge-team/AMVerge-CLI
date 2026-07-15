from __future__ import annotations

import json
import os
import signal
import threading
from pathlib import Path
from typing import Optional

import typer

from ...core.infra.binaries import get_ffmpeg, get_ffprobe
from ...core.infra.ipc import emit_progress, emit_event, log
from ...core.codec.codec_utils import (
    VALID_CODECS, VALID_AUDIO, VALID_CONTAINERS, VALID_HARDWARE, CODEC_ALIASES,
)
from ...core.export import export_scenes, ExportJob, ExportSettings
from ...core.export import params as xparams
from ...core.export.engine import probe_audio_codec
from ...ui import banner, console, make_progress, ok, fail


def _parse_select(select: Optional[str], max_idx: int) -> set[int]:
    if not select:
        return set(range(max_idx + 1))
    picked: set[int] = set()
    for part in select.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            picked.update(range(int(a), int(b) + 1))
        else:
            picked.add(int(part))
    return picked


def _build_jobs(scenes: list[dict], video: Path) -> list[ExportJob]:
    """Resolve each scene's export input: the pre-cut clip file when present,
    else a [start, end] range cut from the source episode (webp mode)."""
    jobs: list[ExportJob] = []
    for s in scenes:
        idx = s["scene_index"]
        clip_path = s.get("clip_path")
        if clip_path and s.get("clip_mode") != "failed" and os.path.exists(clip_path):
            jobs.append(ExportJob(scene_index=idx, input=clip_path))
            continue
        start = s.get("start_sec")
        end = s.get("end_sec")
        seek_ms = int(round(start * 1000)) if isinstance(start, (int, float)) else None
        dur_ms = (
            int(round((end - start) * 1000))
            if isinstance(start, (int, float)) and isinstance(end, (int, float))
            else None
        )
        jobs.append(ExportJob(scene_index=idx, input=str(video), seek_ms=seek_ms, dur_ms=dur_ms))
    return jobs


def export(
    video: Optional[Path] = typer.Argument(None, help="Source video file (with --scenes)"),
    scenes: Optional[Path] = typer.Option(None, "--scenes", "-s", help="scenes/manifest JSON from detect"),
    inputs_json: Optional[Path] = typer.Option(None, "--inputs-json", hidden=True, help="JSON array of input clip paths (app mode)"),
    output: Path = typer.Option(Path("export"), "--output", "-o", help="Output directory"),
    name: Optional[str] = typer.Option(None, "--name", help="Output file stem (default: video name)"),
    select: Optional[str] = typer.Option(None, "--select", help='Indices: "0,2,5-8" (default: all)'),
    merge: bool = typer.Option(False, "--merge", help="Merge selected clips into one file"),
    codec: str = typer.Option("copy", "--codec", help="copy · h264_* · h265_* · av1_main · prores_*"),
    audio: str = typer.Option("copy", "--audio", help="copy · aac · aac_320 · pcm16 · pcm24 · flac · alac · opus · mp3 · none"),
    container: str = typer.Option("mp4", "--container", help="mp4 · mkv · mov"),
    hardware: str = typer.Option("auto", "--hardware", help="auto · gpu · cpu"),
    workers: int = typer.Option(1, "--workers", help="Parallel clip exports"),
    audio_track: int = typer.Option(-1, "--audio-track", help="0-based audio index to hoist to first (preview language); -1 = keep order"),
    ipc: bool = typer.Option(False, "--ipc", hidden=True, help="Emit IPC events for the Tauri app (no Rich UI)"),
) -> None:
    """Export selected scenes from a detect run."""
    if codec not in VALID_CODECS:
        fail(f"Unknown codec '{codec}'. Valid: {', '.join(sorted(VALID_CODECS))}")
        raise typer.Exit(1)
    if audio not in VALID_AUDIO:
        fail(f"Unknown audio '{audio}'. Valid: {', '.join(sorted(VALID_AUDIO))}")
        raise typer.Exit(1)
    if container not in VALID_CONTAINERS:
        fail(f"Unknown container '{container}'. Valid: {', '.join(sorted(VALID_CONTAINERS))}")
        raise typer.Exit(1)
    if hardware not in VALID_HARDWARE:
        fail(f"Unknown hardware '{hardware}'. Valid: {', '.join(sorted(VALID_HARDWARE))}")
        raise typer.Exit(1)

    codec = CODEC_ALIASES.get(codec, codec)
    if codec != "copy" and not xparams.codec_container_compatible(codec, container):
        rec = xparams.recommended_container(codec)
        fail(f"Codec '{codec}' is not compatible with container '{container}'. Use '{rec}'.")
        raise typer.Exit(1)

    output.mkdir(parents=True, exist_ok=True)
    ff, fp = get_ffmpeg(), get_ffprobe()

    if inputs_json is not None:
        # App mode. Each entry is either a string (a pre-cut clip file, exported
        # whole — video mode) or an object with an optional [start_sec, end_sec]
        # range to cut from a source episode (webp mode, no clip file on disk).
        raw = json.loads(inputs_json.read_text(encoding="utf-8"))
        if not raw:
            fail("No input clips provided.")
            raise typer.Exit(1)
        jobs = []
        for i, item in enumerate(raw):
            if isinstance(item, str):
                jobs.append(ExportJob(scene_index=i, input=item))
                continue
            inp = item.get("input")
            if not inp:
                continue
            start, end = item.get("start_sec"), item.get("end_sec")
            seek_ms = int(round(start * 1000)) if isinstance(start, (int, float)) else None
            dur_ms = (
                int(round((end - start) * 1000))
                if isinstance(start, (int, float)) and isinstance(end, (int, float))
                else None
            )
            jobs.append(ExportJob(scene_index=int(item.get("scene_index", i)),
                                  input=str(inp), seek_ms=seek_ms, dur_ms=dur_ms))
        if not jobs:
            fail("No valid input clips provided.")
            raise typer.Exit(1)
    else:
        if video is None or scenes is None:
            fail("Provide VIDEO and --scenes, or --inputs-json.")
            raise typer.Exit(1)
        if not video.exists() or not scenes.exists():
            fail("VIDEO and --scenes must both exist.")
            raise typer.Exit(1)
        payload = json.loads(scenes.read_text(encoding="utf-8"))
        all_scenes = payload.get("scenes", payload) if isinstance(payload, dict) else payload
        if not all_scenes:
            fail("No scenes in JSON.")
            raise typer.Exit(1)
        for s in all_scenes:
            if "scene_index" not in s and "index" in s:
                s["scene_index"] = s["index"]
        max_idx = max(s["scene_index"] for s in all_scenes)
        wanted = _parse_select(select, max_idx)
        selected = [s for s in all_scenes if s["scene_index"] in wanted]
        if not selected:
            fail("No scenes matched selection.")
            raise typer.Exit(1)
        jobs = _build_jobs(selected, video)

    # if 'copy' would fail the muxer (e.g. Opus to mp4), switch to a container-safe encoder up front.
    if audio == "copy":
        acodec = probe_audio_codec(fp, jobs[0].input)
        if acodec and not xparams.audio_copy_safe(acodec, container):
            audio = xparams.fallback_audio_mode(container)
            log(f"audio '{acodec}' not stream-copyable into {container}; using '{audio}'")

    file_stem = name or (video.stem if video else "export")
    settings = ExportSettings(
        codec=codec, audio=audio, container=container,
        hardware=hardware, merge=merge, workers=max(1, workers),
        audio_track=(audio_track if audio_track >= 0 else None),
    )

    abort = threading.Event()

    def _on_signal(_signum, _frame):
        abort.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _on_signal)
        except (ValueError, OSError):
            pass  # not on main thread / unsupported platform

    if ipc:
        try:
            outputs = export_scenes(
                jobs, str(output), file_stem, settings,
                on_progress=emit_progress, on_event=emit_event,
                abort=abort, ffmpeg=ff, ffprobe=fp,
            )
        except Exception as e:  # if BLE001 report structured error to the app
            log(f"EXPORT FATAL: {e}")
            print(json.dumps({"schema_version": "1.0", "outputs": [],
                              "error": {"message": str(e), "type": type(e).__name__}}), flush=True)
            raise typer.Exit(1)
        print(json.dumps({"schema_version": "1.0", "outputs": outputs, "error": None}), flush=True)
        return

    banner("export")
    with make_progress() as progress:
        task = progress.add_task(f"Exporting {len(jobs)} clip(s)", total=100)

        def _cb(pct: int, msg: str) -> None:
            progress.update(task, completed=pct, description=msg)

        outputs = export_scenes(
            jobs, str(output), file_stem, settings,
            on_progress=_cb, abort=abort, ffmpeg=ff, ffprobe=fp,
        )
    ok(f"{len(outputs)} file(s) -> {output}")
