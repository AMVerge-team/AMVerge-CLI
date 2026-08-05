"""Shared Rich UI components styled to match the AMVerge app.

Color palette (from frontend/src/styles/variables.css):
  accent       #22c55e  - main green
  accent.bright #00f07a  - bright green (hover/active states)
  muted        dim grey  - secondary text
  bg           #001a00   - app background tint (not applicable in terminal)
"""
from __future__ import annotations

from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.rule import Rule
from rich.table import Table
from rich.theme import Theme
from rich import box

from ..__version__ import __version__

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------

THEME = Theme(
    {
        "accent":        "#22c55e",
        "accent.bright": "#00f07a",
        "muted":         "bright_black",
        "success":       "#22c55e bold",
        "warn":          "#facc15",
        "error":         "#ef4444",
        "label":         "white",
        "bar.back":      "bright_black",
        "bar.complete":  "#22c55e",
        "bar.finished":  "#00f07a",
    },
    inherit=False,
)

console = Console(theme=THEME, highlight=False)
err     = Console(theme=THEME, stderr=True, highlight=False)

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

def banner(command: str) -> None:
    """Print the AMVerge CLI header panel with the active command name."""
    err.print()
    err.print(
        Panel(
            f"[accent]AMV[/][white bold]erge[/]  [muted]CLI[/]"
            f"  [muted]v{__version__}[/]  [muted]·[/]  [accent]{command}[/]",
            border_style="#22c55e",
            padding=(0, 2),
            expand=False,
        )
    )
    err.print()

# ---------------------------------------------------------------------------
# Progress factory
# ---------------------------------------------------------------------------

def make_progress(
    show_count: bool = False,
    transient: bool = True,
    on_stderr: bool = True,
) -> Progress:
    """Return a themed Rich Progress context manager."""
    extra = [MofNCompleteColumn()] if show_count else []
    return Progress(
        SpinnerColumn(style="accent"),
        TextColumn("[label]{task.description}"),
        BarColumn(
            bar_width=28,
            style="bar.back",
            complete_style="bar.complete",
            finished_style="bar.finished",
        ),
        TaskProgressColumn(style="muted"),
        TimeElapsedColumn(),
        *extra,
        console=err if on_stderr else console,
        transient=transient,
    )

# ---------------------------------------------------------------------------
# Table factory
# ---------------------------------------------------------------------------

def make_table(*columns: tuple[str, str | None, dict], title: str | None = None) -> Table:
    """Return an AMVerge-styled Rich Table.

    Each entry in `columns` is ``(header, style, kwargs)`` where kwargs are
    passed to ``Table.add_column``.
    """
    t = Table(
        title=title,
        box=box.SIMPLE,
        show_edge=False,
        header_style="#22c55e bold",
        border_style="bright_black",
        title_style="#22c55e bold",
    )
    for header, style, kw in columns:
        t.add_column(header, style=style or "", **kw)
    return t

# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def ok(msg: str) -> None:
    """Print a success line to stdout."""
    console.print(f"[accent]>[/] {msg}")

def warn(msg: str) -> None:
    console.print(f"[warn]>[/] {msg}")

def fail(msg: str) -> None:
    # Escape: messages carry literal brackets (ffmpeg filter labels, extras
    # like amverge[ml]) that rich would otherwise strip as style tags.
    err.print(f"[error]>[/] {escape(str(msg))}")

def dim(msg: str) -> None:
    console.print(f"[muted]{msg}[/]")


def gpu_line(indent: str = "  ", label: str = "GPU:", torch_path: bool = True,
             alternatives: str = "") -> dict:
    """Print the GPU status row and return the dict from get_gpu_info().

    Reports the GPU that is physically present, which is not the same question
    as whether PyTorch can drive it. An AMD card is named and shown, then the
    caller's alternatives are offered when the torch path would fall to CPU.

    Args:
        indent: leading whitespace, to match the caller's block.
        label: row label, padded by the caller to align its own rows.
        torch_path: whether the caller is about to run a PyTorch model. Those
            reach the GPU through CUDA only, so a non-NVIDIA card means CPU
            speed. Pass False from paths that run on any vendor, so they stay
            quiet instead of warning about a slowdown that will not happen.
        alternatives: what to suggest when the torch path would be slow,
            phrased for this command. Escaped, so literal brackets survive.
    """
    from ..core.infra.device import torch_backend_gap
    from ..core.infra.diagnostics import get_gpu_info

    info = get_gpu_info()
    name = info.get("gpu_name")
    vram = info.get("vram_gb") or 0.0

    if name:
        vram_txt = f"  VRAM: [accent]{vram:.1f} GB[/]" if vram else ""
        console.print(f"{indent}{label} [accent]{escape(name)}[/]{vram_txt}")

    if not torch_path:
        return info

    gap = torch_backend_gap()
    if gap is None:
        return info

    if gap == "torch_not_cuda":
        msg = ("PyTorch is installed without CUDA, so this runs on CPU. "
               "Reinstall it from https://pytorch.org/get-started/locally/")
    elif gap == "no_torch_backend":
        msg = "PyTorch cannot use this GPU, so this runs on CPU and will be slow."
        if alternatives:
            msg += f" {alternatives}"
    else:
        msg = "No GPU detected. This runs on CPU and will be very slow."

    console.print(f"{indent}[warn]{escape(msg)}[/]")
    return info
