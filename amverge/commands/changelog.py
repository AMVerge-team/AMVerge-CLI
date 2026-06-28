from __future__ import annotations

from ..ui import banner, console, make_table


_ENTRIES = [
    ("v1.2.6", ["Fixed AMVerge updater failing"]),
    ("v1.2.5", ["Fixed videos not playing in Windows Media Player"]),
    ("v1.2.4", [
        "Fixed files with % or special characters in name not importing",
        "Export now sets selected audio stream as default track",
    ]),
    ("v1.2.3", ["Added safeguards to episode clear so it doesn't wipe everything"]),
    ("v1.2.2", [
        "Fixed episodes disappearing on startup",
        "Fixed Python build errors for some Windows users",
    ]),
    ("v1.2.1", ["Fixed hovered videos sometimes not showing full clip content"]),
    ("v1.2.0", [
        "Added audio stream switching for previewing",
        "Added 'Update Available!' in-app notification",
        "Fixed timeline click not working",
        "Fixed audio toggle resetting video",
        "Fixed Intel Macs not importing properly",
    ]),
    ("v1.0.0", [
        "macOS support",
        "Backend merges clips with similar thumbnails to fix awkward cuts",
        "Export profiles with customizable icons",
        "Quick download buttons per clip",
        "Audio hover - plays audio when hovering clips",
        "Discord Rich Presence support",
        "General settings: change episode storage path, reset to defaults",
        "Appearance: GIF background support, built-in cropper, accent → bg sync",
        "Widescreen clip tiles and timestamp toggles",
        "Fixed large video files not importing",
        "Fixed 4K images turning white on import",
    ]),
]


def changelog() -> None:
    """Show AMVerge version history."""
    banner("changelog")

    for version, changes in _ENTRIES:
        t = make_table(
            ("",  "bright_black", {}),
            title=version,
        )
        for c in changes:
            t.add_row(c)
        console.print(t)
        console.print()
