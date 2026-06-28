from __future__ import annotations

from rich.panel import Panel
from rich.table import Table
from rich import box

from ..ui import banner, console


_CLI_TEAM = [
    ("Moongetsu",  "CLI author · package design · interactive wizard"),
]

_APP_TEAM = [
    ("Crptk",          "App owner · developer · original creator"),
    ("Netsuma",         "Export settings · UI upgrades"),
    ("Moongetsu",       "Settings overhaul · Discord RPC · menu revamp"),
    ("Lewis",           "Mac support · background import · heavy optimization"),
    ("0xkhaosoccured",  "Grid UI fixes"),
    ("TOSINIRL",        "Mac video import fixes"),
]


def _credits_table() -> Table:
    t = Table(
        box=box.SIMPLE,
        show_edge=False,
        show_header=False,
        border_style="bright_black",
        padding=(0, 2),
    )
    t.add_column("name", width=18, style="#22c55e bold", no_wrap=True)
    t.add_column("role", style="bright_black")

    t.add_row("[white bold]AMVerge CLI[/]", "", end_section=True)
    for name, role in _CLI_TEAM:
        t.add_row(name, role)

    t.add_row("", "", end_section=True)
    t.add_row("[white bold]AMVerge App[/]", "", end_section=True)
    for name, role in _APP_TEAM:
        t.add_row(name, role)

    return t


def credits() -> None:
    """Show the AMVerge contributors."""
    banner("credits")

    console.print("  [bright_black]The people behind AMVerge.[/]\n")
    console.print(_credits_table())
    console.print()
    console.print(
        Panel(
            "[bright_black]Want to contribute?[/]  [white]github.com/crptk/AMVerge[/]",
            border_style="bright_black",
            padding=(0, 2),
            expand=False,
        )
    )
    console.print()
