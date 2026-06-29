from __future__ import annotations

from ...ui import banner, console, make_table


_CLI_ENTRIES = [
    ("v0.2.0", [
        "V2 backend migration: TransNetV2 ML scene detection (GPU/CPU)",
        "Smart cut: lossless copy + smartcut + re-encode (HEVC snapped-copy on CPU)",
        "Scene cache: .npy files keyed by file fingerprint, skips re-detection on re-open",
        "Streaming IPC: INITIAL_CLIPS_READY, CLIP_READY, PHASE1_COMPLETE, REENCODE_PROGRESS",
        "Nelux Windows native video reader support (optional)",
        "New commands: probe, cache, keyframes, scenes",
        "rpc-server sidecar: long-lived Discord RPC server driven via stdin JSON",
        "Discord RPC: added update_selecting and update_navigating",
        "pyproject.toml: added [ml] extra (torch, transnetv2-pytorch, tqdm)",
    ]),
    ("v0.1.0", [
        "Initial release: keyframe and edge detection methods",
        "amverge detect, export, merge, info commands",
        "Interactive wizard (no-args mode)",
        "IPC mode for Tauri sidecar replacement (--ipc flag)",
        "Streaming thumbnails with THUMBNAIL_READY events",
        "Discord RPC via pypresence (optional [discord] extra)",
        "amverge backend hidden command: drop-in for python app.py",
        "Python library: from amverge import detect_scenes",
    ]),
]

_APP_ENTRIES = [
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
        "Appearance: GIF background support, built-in cropper, accent to bg sync",
        "Widescreen clip tiles and timestamp toggles",
        "Fixed large video files not importing",
        "Fixed 4K images turning white on import",
    ]),
]


def changelog() -> None:
    """Show AMVerge CLI and app version history."""
    banner("changelog")

    console.print("[muted]  CLI[/]\n")
    for version, changes in _CLI_ENTRIES:
        t = make_table(("", "bright_black", {}), title=version)
        for c in changes:
            t.add_row(c)
        console.print(t)
        console.print()

    console.print("[muted]  AMVerge App[/]\n")
    for version, changes in _APP_ENTRIES:
        t = make_table(("", "bright_black", {}), title=version)
        for c in changes:
            t.add_row(c)
        console.print(t)
        console.print()
