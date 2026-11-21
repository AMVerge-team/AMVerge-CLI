from __future__ import annotations

from rich.panel import Panel
from rich.table import Table
from rich import box

from ...ui import banner, console


_CLI_TEAM = [
    ("Moongetsu",  "CLI author · package design · interactive wizard"),
    ("Crptk",      "CLI co-author · app integration · releases"),
    ("Haim Faraj", "Dedup engine · Anime4K & ArtCNN upscale chains"),
    ("Lewis",      "Copy/reencode cutting engine · MPS fallback · test suite"),
    ("Mark-Shun",  "Scene Scout daemon · thumbnails · database tooling"),
]

_APP_TEAM = [
    ("Crptk",          "App owner · developer · original creator"),
    ("Moongetsu",      "Settings overhaul · Discord RPC · menu revamp"),
    ("Haim Faraj",     "French localization · animated icon set · CJK font subsetting"),
    ("Mark-Shun",      "Scene Scout UI · thumbnail extraction · database management"),
    ("Lewis",          "Mac support & DMG installers · background import · optimization"),
    ("robhbzhang",     "Simplified Chinese · macOS native title bar · CLI discovery"),
    ("赵芷若",          "UI & preview polish · clip timestamps · post-export toggle"),
    ("Netsuma",        "Custom export profiles · clip download UI"),
    ("nuruodo",        "Windows install retry fixes · companion JPG poster fix"),
    ("TOSINIRL",       "Mac video import fixes"),
    ("0xkhaosoccured", "Clip grid width fix · larger generated covers"),
]

_EXTENSION_TEAM = [
    ("Crptk",     "Extension owner · panel & AE integration"),
    ("Lewis",     "Episode library home · macOS support · process management"),
    ("Moongetsu", "Auto-update · source pill sync"),
]

# third-party projects AMVerge builds on, credited by their authors
_BUILT_WITH = [
    ("Scene Scout", "scene search · Mark-Shun (Sonicfreak1111) · GPL-3.0"),
    ("Practical-RIFE", "RIFE optical-flow warp · hzwer / Megvii Research · MIT"),
    ("NeLux",  "NVDEC GPU decode · NevermindNilas (orig. Trentonom0r3) · AGPL-3.0"),
    ("Depth Anything", "vendored model code · Apache-2.0"),
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

    t.add_row("", "", end_section=True)
    t.add_row("[white bold]AMVerge Extension[/]", "", end_section=True)
    for name, role in _EXTENSION_TEAM:
        t.add_row(name, role)

    t.add_row("", "", end_section=True)
    t.add_row("[white bold]Built with[/]", "", end_section=True)
    for name, role in _BUILT_WITH:
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
