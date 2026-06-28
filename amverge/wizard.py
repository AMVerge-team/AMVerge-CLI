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
    hint = f" [muted]\[{default}][/]" if default else ""
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
    err.print(f"  [accent]›[/]  [label]{label}[/]  {opts}  [muted]\[{default}][/]")
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
    method = _ask_choice("method", ["keyframe", "edge"], "keyframe")
    err.print(f"  [muted]  keyframe - fast, cuts at I-frame boundaries[/]")
    err.print(f"  [muted]  edge     - accurate, needs opencv  [pip install amverge-cli[edge]][/]\n")
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
    dim(f"scenes.json → {result.scenes_json}")


def _wizard_export() -> None:
    from .core.binaries import get_ffmpeg

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
    codec = _ask_choice("codec", ["copy", "h264", "hevc"], "copy")
    err.print()

    _section("review", "04/04")
    _summary_panel([
        ("scenes",   f"{len(selected)} selected"),
        ("output",   output or "export"),
        ("merge",    "yes" if merge else "no"),
        ("codec",    codec),
    ])

    if not _ask_yn("run export", True):
        return

    err.print()
    output_path = Path(output or "export")
    output_path.mkdir(parents=True, exist_ok=True)
    ff = get_ffmpeg()
    CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0
    import subprocess

    if merge:
        with make_progress() as progress:
            task = progress.add_task(f"Merging {len(selected)} clips", total=1)
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                cfile = f.name
                for s in selected:
                    f.write(f"file '{s['path'].replace(chr(92), '/')}'\n")
            dst = str(output_path / "merged.mp4")
            try:
                cmd = [ff, "-y", "-f", "concat", "-safe", "0", "-i", cfile]
                cmd += ["-c", "copy"] if codec == "copy" else ["-c:v", codec, "-c:a", "aac"]
                cmd.append(dst)
                subprocess.run(cmd, capture_output=True, creationflags=CREATE_NO_WINDOW, check=True)
            finally:
                os.unlink(cfile)
            progress.update(task, completed=1)
        ok(f"Merged → {dst}")
    else:
        with make_progress(show_count=True) as progress:
            task = progress.add_task(f"Exporting {len(selected)} clips", total=len(selected))
            for s in selected:
                idx = s["scene_index"]
                dst = str(output_path / f"scene_{idx:04d}.mp4")
                if codec == "copy":
                    subprocess.run([ff, "-y", "-i", s["path"], "-c", "copy", dst],
                                   capture_output=True, creationflags=CREATE_NO_WINDOW, check=True)
                else:
                    subprocess.run([ff, "-y", "-i", s["path"], "-c:v", codec, "-c:a", "aac", dst],
                                   capture_output=True, creationflags=CREATE_NO_WINDOW, check=True)
                progress.advance(task)
        ok(f"{len(selected)} clips → {output_path}")


def _wizard_merge() -> None:
    from .core.binaries import get_ffmpeg
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

    ok(f"{len(clips)} clips → {out}")


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


# ---------------------------------------------------------------------------
# Info pages
# ---------------------------------------------------------------------------

def _wizard_help() -> None:
    os.system("cls" if os.name == "nt" else "clear")
    from .commands.usage import usage
    usage()


def _wizard_about() -> None:
    os.system("cls" if os.name == "nt" else "clear")
    from .commands.about import about
    about()


def _wizard_credits() -> None:
    os.system("cls" if os.name == "nt" else "clear")
    from .commands.credits import credits
    credits()


def _wizard_changelog() -> None:
    os.system("cls" if os.name == "nt" else "clear")
    from .commands.changelog import changelog
    changelog()


# ---------------------------------------------------------------------------
# Main menu + session loop
# ---------------------------------------------------------------------------

_WORKFLOW: list[tuple[str, str, object]] = [
    ("detect",    "split video into scenes at cut boundaries", _wizard_detect),
    ("export",    "export selected scenes from a detect run",  _wizard_export),
    ("merge",     "merge multiple clips into one file",        _wizard_merge),
    ("info",      "show video stream metadata",                _wizard_info),
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
