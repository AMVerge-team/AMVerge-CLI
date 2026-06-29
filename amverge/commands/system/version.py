from __future__ import annotations

import json
import platform
import sys

import typer

from ...ui import banner, console, make_table
from ...__version__ import __version__


def version(
    as_json: bool = typer.Option(False, "--json", help="Output raw JSON"),
) -> None:
    """Show version info for CLI, Python, and all dependencies."""
    banner("version")

    info: dict = {
        "amverge": __version__,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": platform.system(),
        "platform_version": platform.version(),
        "arch": platform.machine(),
        "deps": {},
    }

    deps = [
        ("av",      "av"),
        ("numpy",   "numpy"),
        ("pillow",  "PIL"),
        ("rich",    "rich"),
        ("typer",   "typer"),
    ]
    optional = [
        ("torch",               "torch"),
        ("transnetv2-pytorch",  "transnetv2_pytorch"),
        ("tqdm",                "tqdm"),
        ("opencv",              "cv2"),
        ("pypresence",          "pypresence"),
    ]

    for name, imp in deps + optional:
        try:
            mod = __import__(imp)
            info["deps"][name] = getattr(mod, "__version__", "installed")
        except ImportError:
            info["deps"][name] = None

    if as_json:
        print(json.dumps(info, indent=2))
        return

    t = make_table(
        ("", "muted",  {"width": 18, "no_wrap": True}),
        ("", "accent", {}),
        title="AMVerge CLI",
    )
    t.add_row("version",          __version__)
    t.add_row("python",           info["python"])
    t.add_row("platform",         f"{info['platform']} {info['arch']}")
    console.print(t)

    t2 = make_table(
        ("", "muted",  {"width": 22, "no_wrap": True}),
        ("", "label",  {}),
        title="Dependencies",
    )
    for name, _ in deps:
        ver = info["deps"].get(name)
        t2.add_row(name, f"v{ver}" if ver else "[error]missing[/]")
    console.print(t2)

    t3 = make_table(
        ("", "muted",  {"width": 22, "no_wrap": True}),
        ("", "label",  {}),
        title="Optional",
    )
    for name, _ in optional:
        ver = info["deps"].get(name)
        t3.add_row(name, f"v{ver}" if ver else "[muted]not installed[/]")
    console.print(t3)
