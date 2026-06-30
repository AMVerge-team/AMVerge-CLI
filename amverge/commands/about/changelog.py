from __future__ import annotations

from ...ui import banner, console, make_table


_CLI_ENTRIES = [
    ("v0.2.0", [
        "AI upscaling: ML models via spandrel (RealCUGAN, Real-ESRGAN)",
        "Anime4K: FFmpeg filter pipeline (lanczos + unsharp + smartblur)",
        "ArtCNN: ONNX Runtime inference (luma-only, 1-channel)",
        "Registry system: registry.json defines all models, CLI auto-discovers",
        "New commands: upscale, models",
        "System monitor: live GPU/CPU/RAM/ETA during upscale",
        "Portable FFmpeg auto-download to %APPDATA%/com.amverge.cli/",
        "9 upscale models: adore, shufflecugan, fallin_soft, fallin_strong, anime4k, C4F16, C4F32, R8F64, realesrgan-x2/x4/anime",
        "Library API: upscale_model(), SystemMonitor",
        "pyproject.toml: \\[upscale\\] extra (torch, opencv, spandrel, onnxruntime)",
    ]),
    ("v0.1.0", [
        "Initial release: keyframe and edge detection methods",
        "amverge detect, export, merge, info commands",
        "Interactive wizard (no-args mode)",
        "IPC mode for Tauri sidecar replacement",
        "Streaming thumbnails with THUMBNAIL_READY events",
        "Discord RPC via pypresence (optional \\[discord\\] extra)",
        "amverge backend hidden command: drop-in for python app.py",
        "Python library: from amverge import detect_scenes",
    ]),
]


def changelog() -> None:
    """Show AMVerge CLI version history."""
    banner("changelog")

    for version, changes in _CLI_ENTRIES:
        t = make_table(
            ("", "bright_black", {}),
            ("", "bright_black", {}),
            title=version,
        )
        for i, c in enumerate(changes, 1):
            t.add_row(f"[muted]{i}.[/]", c)
        console.print(t)
        console.print()
