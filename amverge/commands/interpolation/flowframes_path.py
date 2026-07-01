from __future__ import annotations

from pathlib import Path

import typer

from ...ui import console, fail, ok
from ...core.interpolation import (
    get_flowframes_path,
    set_flowframes_path,
    flowframes_available,
)


def flowframes_path(
    path: Path = typer.Argument(None, help="Path to Flowframes.exe (omit to show current)"),
) -> None:
    """Show or set the Flowframes.exe path for interpolation.

    Without arguments, displays the current configured path.
    With a path argument, sets and persists it for all future runs.
    Currently supports Flowframes 1.42.0 Patreon. Free 1.36.0 support planned (delivery TBD).
    """
    if path is None:
        current = get_flowframes_path()
        if current:
            ok(f"Flowframes path: {current}")
        else:
            console.print(
                "  [warn]No Flowframes path configured.[/warn]\n"
                "  [muted]Auto-detection target: %LOCALAPPDATA%\\Flowframes\\Flowframes.exe[/]\n"
                "  [muted]Set with: amverge flowframes-path PATH[/]"
            )
        return

    if not path.exists():
        fail(f"Not found: {path}")
        raise typer.Exit(1)

    set_flowframes_path(str(path))
    ok(f"Flowframes path saved: {path}")
