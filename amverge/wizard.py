"""Interactive wizard session - launched when amverge is run with no arguments."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.rule import Rule
from rich.table import Table

from .ui import console, err, make_progress, make_table, ok, fail, dim
from .__version__ import __version__


# ---------------------------------------------------------------------------
# Low-level input helpers
# ---------------------------------------------------------------------------

def _ask(label: str, default: str = "", password: bool = False) -> str:
    hint = f" [muted]\\[{default}][/]" if default else ""
    try:
        val = Prompt.ask(
            f"  [accent]›[/]  [label]{label}[/]{hint}",
            console=err,
            default=default,
            password=password,
        )
    except (EOFError, KeyboardInterrupt):
        err.print()
        raise KeyboardInterrupt
    return (val or default).strip()


def _ask_path(label: str, default: str = "", must_exist: bool = False) -> str:
    while True:
        raw = _ask(label, default)
        if must_exist and raw and not Path(raw).exists():
            err.print(f"  [error]  not found:[/] {raw}")
            continue
        return raw


def _ask_yn(label: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    raw = _ask(label, hint).lower()
    return raw in ("y", "yes", "1", "true", hint.lower())


def _ask_float(label: str, default: float, lo: float, hi: float) -> float:
    while True:
        raw = _ask(label, str(default))
        try:
            v = float(raw)
            if lo <= v <= hi:
                return v
        except ValueError:
            pass
        err.print(f"  [error]  enter a number {lo}–{hi}[/]")


def _ask_int(label: str, default: int, lo: int, hi: int) -> int:
    while True:
        raw = _ask(label, str(default))
        try:
            v = int(raw)
            if lo <= v <= hi:
                return v
        except ValueError:
            pass
        err.print(f"  [error]  enter an integer {lo}–{hi}[/]")


def _ask_choice(label: str, choices: list[str], default: str) -> str:
    opts = "  ".join(
        f"[accent bold]{c}[/]" if c == default else f"[muted]{c}[/]"
        for c in choices
    )
    err.print(f"  [accent]›[/]  [label]{label}[/]  {opts}  [muted]\\[{default}][/]")
    while True:
        raw = _ask("", default).lower()
        if raw in choices:
            return raw
        err.print(f"  [error]  choose: {' / '.join(choices)}[/]")


# ---------------------------------------------------------------------------
# Visual helpers
# ---------------------------------------------------------------------------

def _header() -> None:
    os.system("cls" if os.name == "nt" else "clear")
    err.print()
    err.print(
        Panel(
            f"[accent]AMV[/][white bold]erge[/]  [muted]CLI[/]  [muted]v{__version__}[/]",
            border_style="accent",
            padding=(0, 2),
            expand=False,
        )
    )
    err.print()


def _section(title: str, step: str | None = None) -> None:
    step_str = f"[muted]{step}[/]  " if step else ""
    err.print(Rule(f"  {step_str}[accent]{title}[/]  ", style="muted", align="left"))
    err.print()


def _summary_panel(rows: list[tuple[str, str]]) -> None:
    t = Table(box=None, show_header=False, padding=(0, 2))
    t.add_column("key", style="muted", width=14)
    t.add_column("val", style="label")
    for k, v in rows:
        t.add_row(k, v)
    err.print(Panel(t, border_style="muted", title="[muted]summary[/]", title_align="left"))
    err.print()


# ---------------------------------------------------------------------------
# Command wizards
# ---------------------------------------------------------------------------

def _wizard_detect() -> None:
    from .pipeline import detect_scenes, DetectResult

    _header()
    _section("detect", "01/05")
    err.print("  [muted]Split a video into scenes at cut boundaries.[/]\n")

    video = _ask_path("video path", must_exist=True)
    if not video:
        return
    err.print()

    _section("output", "02/05")
    output = _ask_path("output dir  [muted](leave blank for auto)[/]", "")
    output = output or None
    err.print()

    _section("detection", "03/05")
    method = _ask_choice("method", ["keyframe", "edge", "transnetv2"], "keyframe")
    err.print(f"  [muted]  keyframe   - fast, cuts at I-frame boundaries[/]")
    err.print(f"  [muted]  edge       - accurate, needs opencv  [pip install amverge[edge]][/]")
    err.print(f"  [muted]  transnetv2 - ML-based, best accuracy  [pip install amverge[ml]][/]\n")
    min_dur = _ask_float("min scene duration (s)", 0.25, 0.01, 60.0)
    err.print()

    _section("thumbnails & similarity", "04/05")
    thumbs = _ask_yn("generate thumbnails", True)
    similarity = _ask_yn("check scene similarity", True) if thumbs else False
    workers = _ask_int("thumbnail workers", 4, 1, 32) if thumbs else 4
    err.print()

    _section("review", "05/05")
    stem = Path(video).stem
    _summary_panel([
        ("video",      stem),
        ("output",     output or f"{stem}_scenes/"),
        ("method",     method),
        ("min dur",    f"{min_dur}s"),
        ("thumbnails", "yes" if thumbs else "no"),
        ("similarity", "yes" if similarity else "no"),
        ("workers",    str(workers) if thumbs else "-"),
    ])

    if not _ask_yn("run detect", True):
        return

    err.print()
    _LABELS = {
        "detect": "Detecting cuts",
        "segment": "Cutting scenes",
        "thumbnails": "Thumbnails",
        "similarity": "Similarity check",
    }

    with make_progress() as progress:
        tasks: dict[str, object] = {}

        def on_progress(stage: str, pct: int, msg: str) -> None:
            label = _LABELS.get(stage, stage)
            if stage not in tasks:
                tasks[stage] = progress.add_task(label, total=100)
            progress.update(tasks[stage], completed=pct, description=label)

        result: DetectResult = detect_scenes(
            video,
            output_dir=output,
            method=method,
            min_duration=min_dur,
            thumbnails=thumbs,
            similarity=similarity,
            thumbnail_workers=workers,
            progress=on_progress,
        )

    err.print()
    if not result.scenes:
        fail("No scenes detected.")
        return

    similar_set = {idx for pair in result.similar_pairs for idx in pair}
    t = make_table(
        ("#",        "muted", {"justify": "right", "width": 5}),
        ("Start",    None,    {"justify": "right", "width": 9}),
        ("End",      None,    {"justify": "right", "width": 9}),
        ("Duration", None,    {"justify": "right", "width": 9}),
        ("~",        "warn",  {"justify": "center", "width": 3}),
        title=f"{stem}  ·  {len(result.scenes)} scenes  ·  {method}",
    )
    for s in result.scenes:
        t.add_row(
            str(s.index),
            f"{s.start:.2f}s",
            f"{s.end:.2f}s",
            f"{s.duration:.2f}s",
            "~" if s.index in similar_set else "",
        )
    console.print(t)
    ok(f"{len(result.scenes)} scenes  ·  {len(result.similar_pairs)} similar pairs")
    dim(f"scenes.json -> {result.scenes_json}")


def _wizard_export() -> None:
    from .core.infra.binaries import get_ffmpeg

    _header()
    _section("export", "01/04")
    err.print("  [muted]Export selected scenes from a detect run.[/]\n")

    scenes_path = _ask_path("scenes.json path", must_exist=True)
    if not scenes_path:
        return

    payload = json.loads(Path(scenes_path).read_text())
    all_scenes: list[dict] = payload.get("scenes", payload) if isinstance(payload, dict) else payload
    if not all_scenes:
        fail("No scenes in JSON.")
        return

    for s in all_scenes:
        if "scene_index" not in s and "index" in s:
            s["scene_index"] = s["index"]

    max_idx = max(s["scene_index"] for s in all_scenes)
    err.print(f"\n  [muted]{len(all_scenes)} scenes available  (0 – {max_idx})[/]\n")

    _section("selection", "02/04")
    select = _ask("indices  [muted](e.g. 0,2,5-8 or blank for all)[/]", "")
    if select:
        indices: set[int] = set()
        for part in select.split(","):
            part = part.strip()
            if "-" in part:
                lo, hi = part.split("-", 1)
                indices.update(range(int(lo), int(hi) + 1))
            else:
                indices.add(int(part))
        selected = [s for s in all_scenes if s["scene_index"] in indices]
    else:
        selected = all_scenes

    if not selected:
        fail("No scenes matched.")
        return
    err.print()

    _section("options", "03/04")
    output = _ask_path("output dir", "export")
    merge = _ask_yn("merge into one file", False)
    codec = _ask_choice("codec", ["copy", "h264", "hevc", "h264_main", "h265_main", "av1_main", "prores_422"], "copy")
    err.print(f"  [muted]  copy/h264/hevc are aliases for main profiles[/]")
    err.print(f"  [muted]  full list: h264_main/high/high10/high422  h265_main/main10/main12/main422_10  av1_main  prores_*[/]\n")
    audio = _ask_choice("audio", ["copy", "aac", "aac_320", "pcm16", "flac", "opus", "mp3", "none"], "copy")
    container = _ask_choice("container", ["mp4", "mkv", "mov"], "mp4")
    hardware = _ask_choice("hardware", ["auto", "gpu", "cpu"], "auto")
    err.print()

    _section("review", "04/04")
    _summary_panel([
        ("scenes",    f"{len(selected)} selected"),
        ("output",    output or "export"),
        ("merge",     "yes" if merge else "no"),
        ("codec",     codec),
        ("audio",     audio),
        ("container", container),
        ("hardware",  hardware),
    ])

    if not _ask_yn("run export", True):
        return

    err.print()
    output_path = Path(output or "export")
    output_path.mkdir(parents=True, exist_ok=True)
    ff = get_ffmpeg()
    CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
    import subprocess

    from .core.codec.codec_utils import AUDIO_FFMPEG, CODEC_ALIASES, CODEC_PROFILES, PRORES_CODECS, resolve_gpu
    codec = CODEC_ALIASES.get(codec, codec)
    use_gpu = resolve_gpu(hardware, codec)

    if merge:
        with make_progress() as progress:
            task = progress.add_task(f"Merging {len(selected)} clips", total=1)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                cfile = f.name
                for s in selected:
                    f.write(f"file '{s['path'].replace(chr(92), '/')}'\n")
            dst = str(output_path / f"merged.{container}")
            try:
                cmd = [ff, "-y", "-f", "concat", "-safe", "0", "-i", cfile]
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
                os.unlink(cfile)
            progress.update(task, completed=1)
        ok(f"Merged -> {dst}")
    else:
        with make_progress(show_count=True) as progress:
            task = progress.add_task(f"Exporting {len(selected)} clips", total=len(selected))
            for s in selected:
                idx = s["scene_index"]
                dst = str(output_path / f"scene_{idx:04d}.{container}")
                if codec == "copy":
                    cmd = [ff, "-y", "-i", s["path"]]
                    if audio == "copy":
                        cmd += ["-c", "copy"]
                    else:
                        cmd += ["-c:v", "copy"]
                        cmd += AUDIO_FFMPEG[audio]
                    cmd.append(dst)
                else:
                    profile = CODEC_PROFILES[codec]
                    encoder = profile["gpu"] if use_gpu and profile["gpu"] else profile["cpu"]
                    cmd = [ff, "-y", "-i", s["path"], "-c:v", str(encoder)]
                    args = str(profile["args"]).strip()
                    if args:
                        cmd += args.split()
                    cmd += AUDIO_FFMPEG[audio]
                    cmd.append(dst)
                subprocess.run(cmd, capture_output=True, creationflags=CREATE_NO_WINDOW, check=True)
                progress.advance(task)
        ok(f"{len(selected)} clips -> {output_path}")


def _wizard_merge() -> None:
    from .core.infra.binaries import get_ffmpeg
    import subprocess

    _header()
    _section("merge", "01/03")
    err.print("  [muted]Merge multiple clips into one file.[/]\n")
    err.print("  [muted]Enter clip paths one by one. Empty line when done.[/]\n")

    clips: list[Path] = []
    while True:
        raw = _ask(f"clip {len(clips) + 1}  [muted](blank to finish)[/]", "")
        if not raw:
            if len(clips) >= 2:
                break
            err.print("  [error]  need at least 2 clips[/]")
            continue
        p = Path(raw)
        if not p.exists():
            err.print(f"  [error]  not found:[/] {raw}")
            continue
        clips.append(p)
        err.print(f"  [accent]  ✓[/] added  [muted]{p.name}[/]")

    err.print()
    _section("output", "02/03")
    out_raw = _ask("output file", "merged.mp4")
    out = Path(out_raw or "merged.mp4")
    err.print()

    _section("review", "03/03")
    clip_names = "\n".join(f"  [muted]{i+1}.[/] {c.name}" for i, c in enumerate(clips))
    err.print(clip_names)
    _summary_panel([
        ("clips",  str(len(clips))),
        ("output", str(out)),
    ])

    if not _ask_yn("run merge", True):
        return

    err.print()
    out.parent.mkdir(parents=True, exist_ok=True)
    ff = get_ffmpeg()
    CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

    with make_progress() as progress:
        task = progress.add_task(f"Merging {len(clips)} clips", total=1)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            cfile = f.name
            for c in clips:
                f.write(f"file '{str(c.resolve()).replace(chr(92), '/')}'\n")
        try:
            subprocess.run(
                [ff, "-y", "-f", "concat", "-safe", "0", "-i", cfile, "-c", "copy", str(out)],
                capture_output=True, creationflags=CREATE_NO_WINDOW, check=True,
            )
        except subprocess.CalledProcessError as e:
            fail(f"ffmpeg: {e.stderr.decode(errors='replace')[-300:]}")
            return
        finally:
            os.unlink(cfile)
        progress.update(task, completed=1)

    ok(f"{len(clips)} clips -> {out}")


def _wizard_info() -> None:
    from .core.video import get_video_info

    _header()
    _section("info")
    err.print("  [muted]Show video and audio stream metadata.[/]\n")

    video = _ask_path("video path", must_exist=True)
    if not video:
        return

    err.print()
    data = get_video_info(video)
    dur = data["duration"]
    h, m, s = int(dur // 3600), int((dur % 3600) // 60), dur % 60
    dur_str = (f"{h}h {m:02d}m {s:05.2f}s" if h else f"{m}m {s:05.2f}s" if m else f"{s:.2f}s")

    def _br(bps):
        if not bps: return "-"
        return f"{bps/1_000_000:.1f} Mbps" if bps >= 1_000_000 else f"{bps/1_000:.0f} kbps"

    console.print(f"\n[label]{Path(video).name}[/]  [muted]{dur_str}[/]\n")

    for stream in data["streams"]:
        if stream["type"] == "video":
            t = make_table(("", "muted", {"width": 14}), ("", "label", {}), title="Video")
            t.add_row("Codec",      stream["codec"])
            t.add_row("Resolution", f"{stream['width']}×{stream['height']}")
            t.add_row("FPS",        str(stream["fps"]))
            t.add_row("Bitrate",    _br(stream["bit_rate"]))
            console.print(t)
        elif stream["type"] == "audio":
            t = make_table(("", "muted", {"width": 14}), ("", "label", {}), title="Audio")
            t.add_row("Codec",       stream["codec"])
            t.add_row("Sample rate", f"{stream['sample_rate']} Hz")
            t.add_row("Channels",    str(stream["channels"]))
            t.add_row("Bitrate",     _br(stream["bit_rate"]))
            console.print(t)





def _wizard_upscale() -> None:
    os.system("cls" if os.name == "nt" else "clear")
    from .ui import banner, console, err, ok, fail, make_progress
    from .core.upscaling import (
        UPSCALE_AVAILABLE, UPSCALE_REGISTRY,
        upscale_model, download_weights, is_weight_downloaded,
        download_anime4k_shaders, is_anime4k_downloaded,
        download_artcnn, is_artcnn_downloaded,
    )
    banner("upscale")
    if not UPSCALE_AVAILABLE:
        fail("Upscaling not available.\n  pip install amverge[upscale]")
        return
    input_video = _ask("video path", "video.mp4")
    if not input_video or not Path(input_video).exists():
        fail(f"File not found: {input_video}")
        return
    keys = list(UPSCALE_REGISTRY.keys())
    model_key = _ask_choice("model key", keys, keys[0])
    entry = UPSCALE_REGISTRY[model_key]
    method = entry.get("method", "ml")
    if method == "ml" and not is_weight_downloaded(model_key):
        with make_progress() as progress:
            task_id = progress.add_task(f"Downloading {entry.get('name', model_key)}...", total=100)
            download_weights(model_key, progress_cb=lambda p, m: progress.update(task_id, completed=p, description=m))
    elif method == "shader" and not is_anime4k_downloaded():
        with make_progress() as progress:
            task_id = progress.add_task("Downloading Anime4K shaders...", total=100)
            download_anime4k_shaders()
            progress.update(task_id, completed=100, description="Done")
    elif method == "onnx" and not is_artcnn_downloaded(model_key):
        download_artcnn(model_key)
    scale = _ask_int("scale factor", 2, 1, 8)
    output = f"{Path(input_video).stem}_upscaled.mp4"
    with make_progress() as progress:
        task_id = progress.add_task("Upscaling...", total=100)
        try:
            upscale_model(model_key, input_path=input_video, output_path=output, scale=scale,
                          progress_cb=lambda p, m: progress.update(task_id, completed=p, description=m))
            ok(f"Saved: {output}")
        except Exception as e:
            fail(str(e))


def _wizard_interpolate() -> None:
    os.system("cls" if os.name == "nt" else "clear")
    from .ui import banner, console, err, ok, fail, make_progress
    from .core.interpolation import (
        INTERPOLATION_REGISTRY, INTERPOLATION_AVAILABLE,
        interpolate_video, download_weights, is_weight_downloaded,
        flowframes_available, run_flowframes,
    )
    banner("interpolate")
    if not INTERPOLATION_AVAILABLE:
        fail("Interpolation not available.\n  pip install amverge[interpolation]")
        return
    input_video = _ask("video path", "video.mp4")
    if not input_video or not Path(input_video).exists():
        fail(f"File not found: {input_video}")
        return
    use_flowframes = _ask_yn("use Flowframes (external .exe)", False)
    if use_flowframes:
        if not flowframes_available():
            fail("Flowframes not available.\nSet with: amverge flowframes-path <path>")
            return
        factor = _ask_int("interpolation factor", 2, 2, 16)
        output_dir = os.path.dirname(input_video) or "."
        with make_progress() as progress:
            task_id = progress.add_task("Interpolating with Flowframes...", total=100)
            try:
                run_flowframes(input_path=input_video, output_dir=output_dir, factor=factor,
                               progress_cb=lambda p, m: progress.update(task_id, completed=p, description=m))
                ok(f"Done in {output_dir}")
            except Exception as e:
                fail(str(e))
    else:
        keys = list(INTERPOLATION_REGISTRY.keys())
        model_key = _ask_choice("model key", keys, keys[0])
        if not is_weight_downloaded(model_key):
            with make_progress() as progress:
                entry = INTERPOLATION_REGISTRY[model_key]
                task_id = progress.add_task(f"Downloading {entry.get('name', model_key)}...", total=100)
                download_weights(model_key, progress_cb=lambda p, m: progress.update(task_id, completed=p, description=m))
        factor = _ask_int("interpolation factor", 2, 2, 16)
        output = f"{Path(input_video).stem}_{factor}x.mp4"
        with make_progress() as progress:
            task_id = progress.add_task("Interpolating...", total=100)
            try:
                interpolate_video(input_video, output, factor=factor, model_key=model_key,
                                  progress_cb=lambda p, m: progress.update(task_id, completed=p, description=m))
                ok(f"Saved: {output}")
            except Exception as e:
                fail(str(e))


def _wizard_deadframes() -> None:
    os.system("cls" if os.name == "nt" else "clear")
    from .ui import banner, ok, fail, make_progress
    from .core.deadframes import (
        DEADFRAMES_REGISTRY, DEADFRAMES_AVAILABLE,
        run_deadframes, is_weight_downloaded, download_weights,
    )
    banner("deadframes")
    if not DEADFRAMES_AVAILABLE:
        fail("OpenCV not installed.\n  pip install amverge[deadframes]")
        return
    input_video = _ask("video path", "video.mp4")
    if not input_video or not Path(input_video).exists():
        fail(f"File not found: {input_video}")
        return
    keys = list(DEADFRAMES_REGISTRY.keys())
    method_key = _ask_choice("method key", keys, keys[0])
    entry = DEADFRAMES_REGISTRY[method_key]
    if "file" in entry and not is_weight_downloaded(method_key):
        with make_progress() as progress:
            task_id = progress.add_task(
                f"Downloading {entry.get('name', method_key)}...", total=100
            )
            download_weights(
                method_key,
                progress_cb=lambda p, m: progress.update(
                    task_id, completed=p, description=m
                ),
            )
    keep_talking = _ask_yn("keep talking (subtle mouth motion)", False)
    keep_camera = _ask_yn("keep camera pan/zoom/shake", False)
    safe = _ask_yn("safe mode (--keep-talking --keep-camera)", False)
    use_auto = safe or keep_talking or keep_camera or _ask_yn("auto-calibrate thresholds", False)
    cadence = _ask_int("cadence (min consecutive dead to drop)", 3, 1, 30)
    no_audio = _ask_yn("drop audio", False)
    output = f"{Path(input_video).stem}_deadframes.mp4"
    use_keep_talking = keep_talking or safe
    use_keep_camera = keep_camera or safe
    with make_progress() as progress:
        task_id = progress.add_task("Removing dead frames...", total=100)
        try:
            result = run_deadframes(
                input_path=input_video,
                output_path=output,
                keep_talking=use_keep_talking,
                keep_camera=use_keep_camera,
                auto=use_auto,
                cadence=cadence,
                no_audio=no_audio,
                progress_cb=lambda p, m: progress.update(
                    task_id, completed=p, description=m
                ),
            )
            ok(f"Saved: {output} ({result['duration_after']:.1f}s)")
        except Exception as e:
            fail(str(e))


def _wizard_pipeline() -> None:
    os.system("cls" if os.name == "nt" else "clear")
    from .commands.pipeline.pipeline import _PIPELINE_ENABLED, _AVAILABLE_OPS
    if not _PIPELINE_ENABLED:
        from .ui import banner, fail
        banner("pipeline")
        available = [k for k, v in _AVAILABLE_OPS.items() if v]
        missing = [k for k, v in _AVAILABLE_OPS.items() if not v]
        fail(
            f"Pipeline requires at least 2 operations installed.\n"
            f"  Installed: {', '.join(available) if available else 'none'}\n"
            f"  Missing:   {', '.join(missing)}\n"
            f"  Run: pip install amverge[{','.join(missing)}]"
        )
        return
    from .commands.pipeline.pipeline import pipeline as _run_pipeline
    _run_pipeline()


def _wizard_depth_map() -> None:
    os.system("cls" if os.name == "nt" else "clear")
    from .ui import banner, console, err, ok, fail, make_progress
    from .core.depth import (
        DEPTH_AVAILABLE, MODEL_CONFIGS, is_model_downloaded,
        download_model, generate_depth_map,
    )
    banner("depth-map")
    if not DEPTH_AVAILABLE:
        fail("Depth-Anything-V2 not installed.\n  pip install amverge[depth]")
        return
    input_video = _ask("video path", "video.mp4")
    if not input_video or not Path(input_video).exists():
        fail(f"File not found: {input_video}")
        return
    encoder = _ask("encoder (vits/vitb/vitl)", "vits").strip().lower()
    if encoder not in MODEL_CONFIGS:
        fail(f"Unknown encoder: {encoder}. Using vits.")
        encoder = "vits"
    pred_only = _ask("depth only? (y/n)", "n").strip().lower().startswith("y")
    grayscale = _ask("grayscale? (y/n)", "n").strip().lower().startswith("y")
    output = f"{Path(input_video).stem}_depth.mp4"
    if not is_model_downloaded(encoder):
        with make_progress() as progress:
            task_id = progress.add_task(f"Downloading Depth-Anything-V2-{encoder}...", total=100)
            download_model(encoder, progress_cb=lambda p, m: progress.update(task_id, completed=p, description=m))
    with make_progress() as progress:
        task_id = progress.add_task("Processing...", total=100)
        try:
            generate_depth_map(
                input_path=input_video,
                output_path=output,
                encoder=encoder,
                pred_only=pred_only,
                grayscale=grayscale,
                progress_cb=lambda p, m: progress.update(task_id, completed=p, description=m),
            )
            ok(f"Saved: {output}")
        except Exception as e:
            fail(str(e))


# ---------------------------------------------------------------------------
# Info pages
# ---------------------------------------------------------------------------

def _wizard_help() -> None:
    os.system("cls" if os.name == "nt" else "clear")
    from .commands.about.usage import usage
    usage()


def _wizard_about() -> None:
    os.system("cls" if os.name == "nt" else "clear")
    from .commands.about.about import about
    about()


def _wizard_credits() -> None:
    os.system("cls" if os.name == "nt" else "clear")
    from .commands.about.credits import credits
    credits()


def _wizard_changelog() -> None:
    os.system("cls" if os.name == "nt" else "clear")
    from .commands.about.changelog import changelog
    changelog()


# ---------------------------------------------------------------------------
# Main menu + session loop
# ---------------------------------------------------------------------------

_WORKFLOW: list[tuple[str, str, object]] = [
    ("detect",      "split video into scenes at cut boundaries", _wizard_detect),
    ("export",      "export selected scenes from a detect run",  _wizard_export),
    ("merge",       "merge multiple clips into one file",        _wizard_merge),
    ("info",        "show video stream metadata",                _wizard_info),
    ("upscale",     "AI upscale video (ml, shader, onnx)",       _wizard_upscale),
    ("interpolate", "frame interpolation (RIFE or Flowframes)",  _wizard_interpolate),
    ("deadframes", "remove static frames (deadframe remover)",     _wizard_deadframes),
    ("pipeline",   "chain deadframes + upscale + interpolate",      _wizard_pipeline),
    ("depth-map",   "generate depth maps from video",            _wizard_depth_map),
]

_INFO: list[tuple[str, str, object]] = [
    ("usage",     "command reference and usage examples",  _wizard_help),
    ("about",     "what is AMVerge CLI",                   _wizard_about),
    ("credits",   "meet the team",                         _wizard_credits),
    ("changelog", "version history",                       _wizard_changelog),
]

_ALL_COMMANDS = _WORKFLOW + _INFO


def _show_menu() -> None:
    _header()

    t = Table(box=None, show_header=False, padding=(0, 2), show_edge=False)
    t.add_column("num",  style="muted",       width=4)
    t.add_column("cmd",  style="#22c55e bold", width=12)
    t.add_column("desc", style="muted")

    for i, (cmd, desc, _) in enumerate(_WORKFLOW, 1):
        t.add_row(f"0{i}", cmd, desc)

    t.add_row("", "", "")

    for i, (cmd, desc, _) in enumerate(_INFO, len(_WORKFLOW) + 1):
        num = f"{i:02d}"
        t.add_row(num, cmd, desc)

    t.add_row("", "", "")
    t.add_row("00", "quit", "exit session")

    err.print(Panel(t, border_style="muted", padding=(0, 1)))
    err.print()


def run_wizard() -> None:
    """Launch the interactive AMVerge CLI session."""
    cmd_map = {f"{i:02d}": fn for i, (_, _, fn) in enumerate(_ALL_COMMANDS, 1)}
    cmd_map.update({str(i): fn for i, (_, _, fn) in enumerate(_ALL_COMMANDS, 1)})
    name_map = {cmd: fn for cmd, _, fn in _ALL_COMMANDS}

    while True:
        _show_menu()

        raw = _ask("command", "").lower().strip()

        if raw in ("00", "0", "quit", "exit", "q", ""):
            err.print("\n  [muted]bye[/]\n")
            break

        fn = cmd_map.get(raw) or name_map.get(raw)
        if fn is None:
            err.print(f"  [error]  unknown command[/] '{raw}'")
            import time; time.sleep(0.8)
            continue

        try:
            fn()
        except KeyboardInterrupt:
            err.print("\n  [muted]cancelled[/]")

        err.print()
        _ask("press enter to continue", "")
