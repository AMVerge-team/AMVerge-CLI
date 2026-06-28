from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import typer

from ..core.binaries import get_ffmpeg
from ..core.discord_rpc import RPC_AVAILABLE, DiscordRPC
from ..core.codec_utils import (
    VALID_CODECS, VALID_AUDIO, VALID_CONTAINERS, VALID_HARDWARE,
    CODEC_ALIASES, CODEC_PROFILES, PRORES_CODECS, AUDIO_FFMPEG,
    _resolve_gpu,
)
from ..ui import banner, console, err, make_progress, make_table, ok, fail, dim

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def export(
    video: Path = typer.Argument(..., help="Source video file", exists=True),
    scenes: Path = typer.Option(..., "--scenes", "-s", help="scenes.json from detect", exists=True),
    output: Path = typer.Option(Path("export"), "--output", "-o", help="Output directory"),
    select: Optional[str] = typer.Option(None, "--select", help='Indices: "0,2,5-8" (default: all)'),
    merge: bool = typer.Option(False, "--merge", help="Merge selected clips into one file"),
    codec: str = typer.Option("copy", "--codec", help="copy · h264 · hevc"),
    audio: str = typer.Option("copy", "--audio", help="copy · aac · aac_320 · pcm16 · pcm24 · flac · alac · opus · mp3 · none"),
    container: str = typer.Option("mp4", "--container", help="mp4 · mkv · mov"),
    hardware: str = typer.Option("auto", "--hardware", help="auto · gpu · cpu"),
    no_rpc: bool = typer.Option(False, "--no-rpc", help="Disable Discord RPC"),
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
    use_gpu = _resolve_gpu(hardware, codec)
    if codec in PRORES_CODECS and container != "mov":
        fail(f"Codec '{codec}' requires --container mov")
        raise typer.Exit(1)

    banner("export")

    rpc = DiscordRPC() if RPC_AVAILABLE and not no_rpc else None
    if rpc:
        rpc.connect()
        rpc.update_exporting(video.name)

    payload = json.loads(scenes.read_text())
    all_scenes: list[dict] = payload.get("scenes", payload) if isinstance(payload, dict) else payload

    if not all_scenes:
        fail("No scenes in JSON.")
        raise typer.Exit(1)

    for s in all_scenes:
        if "scene_index" not in s and "index" in s:
            s["scene_index"] = s["index"]

    max_idx = max(s["scene_index"] for s in all_scenes)
    selected = (
        [s for s in all_scenes if s["scene_index"] in _parse_select(select, max_idx)]
        if select else all_scenes
    )

    if not selected:
        fail("No scenes matched selection.")
        raise typer.Exit(1)

    output.mkdir(parents=True, exist_ok=True)
    ff = get_ffmpeg()

    try:
        if merge:
            _export_merged(selected, output, ff, codec, audio, container, use_gpu)
        else:
            _export_individual(selected, output, ff, codec, audio, container, use_gpu)
        if rpc:
            rpc.update_complete()
    except Exception:
        if rpc:
            rpc.update_error("Export failed")
        raise
    finally:
        if rpc:
            rpc.clear_presence()
            rpc.disconnect()


def _copy(src: str, dst: str, ff: str, audio: str) -> None:
    cmd = [ff, "-y", "-i", src]
    if audio == "copy":
        cmd += ["-c", "copy"]
    else:
        cmd += ["-c:v", "copy"]
        cmd += AUDIO_FFMPEG[audio]
    cmd.append(dst)
    subprocess.run(cmd, capture_output=True, creationflags=CREATE_NO_WINDOW, check=True)


def _encode(src: str, dst: str, ff: str, codec: str, audio: str, use_gpu: bool) -> None:
    profile = CODEC_PROFILES[codec]
    encoder = profile["gpu"] if use_gpu and profile["gpu"] else profile["cpu"]
    cmd = [ff, "-y", "-i", src, "-c:v", str(encoder)]
    args = str(profile["args"]).strip()
    if args:
        cmd += args.split()
    cmd += AUDIO_FFMPEG[audio]
    cmd.append(dst)
    subprocess.run(cmd, capture_output=True, creationflags=CREATE_NO_WINDOW, check=True)


def _export_individual(scenes: list[dict], output: Path, ff: str, codec: str, audio: str, container: str, use_gpu: bool) -> None:
    with make_progress(show_count=True) as progress:
        task = progress.add_task(f"Exporting {len(scenes)} clips", total=len(scenes))
        for s in scenes:
            idx = s["scene_index"]
            dst = str(output / f"scene_{idx:04d}.{container}")
            if codec == "copy":
                _copy(s["path"], dst, ff, audio)
            else:
                _encode(s["path"], dst, ff, codec, audio, use_gpu)
            progress.advance(task)
            progress.update(task, description=f"Exported scene_{idx:04d}")

    ok(f"{len(scenes)} clips -> {output}")


def _export_merged(scenes: list[dict], output: Path, ff: str, codec: str, audio: str, container: str, use_gpu: bool) -> None:
    with make_progress() as progress:
        task = progress.add_task(f"Merging {len(scenes)} clips", total=1)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            concat_file = f.name
            for s in scenes:
                f.write(f"file '{s['path'].replace(chr(92), '/')}'\n")

        dst = str(output / f"merged.{container}")
        try:
            cmd = [ff, "-y", "-f", "concat", "-safe", "0", "-i", concat_file]
            if codec == "copy":
                if audio == "copy":
                    cmd += ["-c", "copy"]
                else:
                    cmd += ["-c:v", "copy"]
                    cmd += AUDIO_FFMPEG[audio]
            else:
                profile = CODEC_PROFILES[codec]
                encoder = profile["gpu"] if use_gpu and profile["gpu"] else profile["cpu"]
                cmd += ["-c:v", str(encoder)]
                args = str(profile["args"]).strip()
                if args:
                    cmd += args.split()
                cmd += AUDIO_FFMPEG[audio]
            cmd.append(dst)
            subprocess.run(cmd, capture_output=True, creationflags=CREATE_NO_WINDOW, check=True)
        finally:
            os.unlink(concat_file)

        progress.update(task, completed=1)

    ok(f"Merged -> {dst}")
