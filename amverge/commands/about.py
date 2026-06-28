from __future__ import annotations

import typer

from ..ui import banner, console
from ..ui import make_table
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
        "Use it to split anime episodes (or any video) into scenes at cut boundaries, "
        "browse the results, export only the clips you want, and merge fragments back "
        "together - all from a terminal or your own Python scripts.\n\n"
        "Built on [accent]FFmpeg[/] and [accent]PyAV[/]. No GUI required."
    )
    console.print(Panel(blurb, border_style="bright_black", padding=(1, 2)))
    console.print()

    t = make_table(
        ("",  "bright_black", {"width": 18}),
        ("",  "white",        {}),
        title="Key features",
    )
    t.add_row("Keyframe detection",  "near-instant splitting using I-frames, no re-encode")
    t.add_row("Edge detection",      "cosine-similarity approach for difficult encodes")
    t.add_row("Thumbnails",          "auto-generated scene previews via PyAV")
    t.add_row("Similarity check",    "flags duplicate or near-identical adjacent scenes")
    t.add_row("Python library",      "from amverge import detect_scenes")
    t.add_row("Zero quality loss",   "copy-mode export keeps the original stream intact")
    console.print(t)

    console.print()
    console.print("[bright_black]  Source  [/][white]github.com/crptk/AMVerge[/]")
    console.print("[bright_black]  Discord [/][white]discord.gg/bmXjTgsAaN[/]")
    console.print()
