from __future__ import annotations

from rich.markup import escape as rich_escape
from rich.panel import Panel

from ...ui import banner, console


_CLI_ENTRIES = [
    ("v0.2.1", [
        "Anime4K: real GLSL shaders via FFmpeg libplacebo (was lanczos approximation)",
        "Anime4K: auto-detect libplacebo, lanczos+unsharp fallback when unavailable",
        "Anime4K: shader chains per mode (light/medium/strong), 2 passes for 4x",
        "Anime4K: strong mode adds Clamp_Highlights (official Mode A chain)",
        "ArtCNN: fixed ONNX path bug (download dir now matches read dir)",
        "ArtCNN: fixed HD out-of-memory (disable onnx arena, free per frame)",
        "ArtCNN: new models R16F96, C4F16_DN, C4F16_DS, C4F32_DN",
        "ArtCNN: chroma reconstruction models R8F64_Chroma, R8F64_Chroma_DN (CNN U/V)",
        "Encode: auto x264 level (was hardcoded 5.1, broke 4x of HD)",
        "Encode: preserve source color metadata (BT.709/range passthrough)",
        "Audio: copy source stream untouched, re-encode to AAC only as fallback",
        "Upscaling: split engine.py into anime4k.py / artcnn.py / ffmpeg_helpers.py",
    ]),
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
        "pyproject.toml: [upscale] extra (torch, opencv, spandrel, onnxruntime)",
    ]),
    ("v0.1.0", [
        "Initial release: keyframe and edge detection methods",
        "amverge detect, export, merge, info commands",
        "Interactive wizard (no-args mode)",
        "IPC mode for Tauri sidecar replacement",
        "Streaming thumbnails",
        "Discord RPC (optional)",
        "amverge backend hidden command",
        "Python library: from amverge import detect_scenes",
    ]),
]


def _render_version(version, changes):
    lines = [f"[bold #22c55e]{version}[/]"]
    lines.append("[dim]----------------------------------------[/]")
    for i, c in enumerate(changes, 1):
        lines.append(f"  [muted]{i:2d}.[/] {rich_escape(c)}")
    return Panel("\n".join(lines), border_style="#22c55e", padding=(1, 2), safe_box=True)


def changelog() -> None:
    """Show AMVerge CLI version history."""
    banner("changelog")
    console.print()
    for version, changes in _CLI_ENTRIES:
        console.print(_render_version(version, changes))
        console.print()


def whatsnew() -> None:
    """Show what's new in the latest version."""
    version, changes = _CLI_ENTRIES[0]
    banner(f"what's new in {version}")
    console.print()
    console.print(_render_version(version, changes))
    console.print()
