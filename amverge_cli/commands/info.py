from __future__ import annotations

from pathlib import Path

import typer

from ..core.video import get_video_info
from ..ui import banner, console, make_table, dim


def _fmt_bitrate(bps: int | None) -> str:
    if not bps:
        return "—"
    if bps >= 1_000_000:
        return f"{bps / 1_000_000:.1f} Mbps"
    return f"{bps / 1_000:.0f} kbps"


def _fmt_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h:
        return f"{h}h {m:02d}m {s:05.2f}s"
    if m:
        return f"{m}m {s:05.2f}s"
    return f"{s:.2f}s"


def info(
    video: Path = typer.Argument(..., help="Video file", exists=True),
) -> None:
    """Show video stream metadata."""
    banner("info")

    data = get_video_info(str(video.resolve()))
    console.print(f"[label]{video.name}[/]  [muted]{_fmt_duration(data['duration'])}[/]\n")

    for stream in data["streams"]:
        if stream["type"] == "video":
            t = make_table(
                ("",      "muted",  {"width": 14, "no_wrap": True}),
                ("",      "label",  {}),
                title="Video",
            )
            t.add_row("Codec",      stream["codec"])
            t.add_row("Resolution", f"{stream['width']}×{stream['height']}")
            t.add_row("FPS",        str(stream["fps"]))
            t.add_row("Bitrate",    _fmt_bitrate(stream["bit_rate"]))
            console.print(t)

        elif stream["type"] == "audio":
            t = make_table(
                ("",      "muted",  {"width": 14, "no_wrap": True}),
                ("",      "label",  {}),
                title="Audio",
            )
            t.add_row("Codec",       stream["codec"])
            t.add_row("Sample rate", f"{stream['sample_rate']} Hz")
            t.add_row("Channels",    str(stream["channels"]))
            t.add_row("Bitrate",     _fmt_bitrate(stream["bit_rate"]))
            console.print(t)
