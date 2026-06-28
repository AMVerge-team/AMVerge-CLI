from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from ..core.video import get_video_info

console = Console()


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
    """Show video metadata."""
    data = get_video_info(str(video.resolve()))

    console.print(f"\n[bold]{video.name}[/bold]  [dim]{_fmt_duration(data['duration'])}[/dim]\n")

    for stream in data["streams"]:
        if stream["type"] == "video":
            t = Table(show_header=False, box=None, padding=(0, 2))
            t.add_column("key", style="dim", width=14)
            t.add_column("val")
            t.add_row("Codec", stream["codec"])
            t.add_row("Resolution", f"{stream['width']}×{stream['height']}")
            t.add_row("FPS", str(stream["fps"]))
            t.add_row("Bitrate", _fmt_bitrate(stream["bit_rate"]))
            console.print("[cyan]Video[/cyan]")
            console.print(t)

        elif stream["type"] == "audio":
            t = Table(show_header=False, box=None, padding=(0, 2))
            t.add_column("key", style="dim", width=14)
            t.add_column("val")
            t.add_row("Codec", stream["codec"])
            t.add_row("Sample rate", f"{stream['sample_rate']} Hz")
            t.add_row("Channels", str(stream["channels"]))
            t.add_row("Bitrate", _fmt_bitrate(stream["bit_rate"]))
            console.print("[cyan]Audio[/cyan]")
            console.print(t)
