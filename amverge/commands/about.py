from __future__ import annotations

import typer

from ..ui import banner, console, make_table
from ..__version__ import __version__
from rich.panel import Panel


def about() -> None:
    """Show what AMVerge CLI is and what it does."""
    banner("about")

    console.print(
        Panel(
            "[accent]AMV[/][white bold]erge[/]  [muted]CLI[/]  [muted]v" + __version__ + "[/]",
            border_style="#22c55e",
            padding=(0, 2),
            expand=False,
        )
    )
    console.print()

    blurb = (
        "AMVerge CLI ports the scene-detection and clip-management engine from the "
        "[accent]AMVerge[/] desktop app into a standalone Python library and CLI tool.\n\n"
        "Use it to split anime episodes (or any video) into scenes using "
        "[accent]TransNetV2[/] ML detection or fast keyframe analysis, "
        "export only the clips you want, and merge fragments - all from a terminal "
        "or your own Python scripts.\n\n"
        "Built on [accent]FFmpeg[/], [accent]PyAV[/], and [accent]PyTorch[/]. No GUI required."
    )
    console.print(Panel(blurb, border_style="bright_black", padding=(1, 2)))
    console.print()

    t = make_table(
        ("", "bright_black", {"width": 20}),
        ("", "white",        {}),
        title="Key features",
    )
    t.add_row("TransNetV2 detection", "ML scene detection, GPU-accelerated, highly accurate")
    t.add_row("Keyframe detection",   "near-instant splitting via I-frames, no ML required")
    t.add_row("Smart cut",            "lossless copy for keyframe-aligned scenes, smartcut or re-encode for the rest")
    t.add_row("HEVC support",         "snapped-copy on CPU, full re-encode with CUDA")
    t.add_row("Scene cache",          "TransNetV2 results saved as .npy - skips re-detection on re-open")
    t.add_row("Streaming IPC",        "CLIP_READY events stream to Tauri as each scene finishes cutting")
    t.add_row("Discord RPC",          "live status via pypresence, same app ID as AMVerge desktop")
    t.add_row("Python library",       "from amverge import detect_scenes")
    t.add_row("Zero quality loss",    "copy-mode export keeps original stream intact")
    console.print(t)

    console.print()
    console.print("[bright_black]  Source  [/][white]github.com/crptk/AMVerge[/]")
    console.print("[bright_black]  Discord [/][white]discord.gg/bmXjTgsAaN[/]")
    console.print()
