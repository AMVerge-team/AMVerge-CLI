from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from ..ui import banner, console, make_table


def _check(label: str, ok: bool, detail: str = "", fix: str = "") -> tuple[str, bool, str, str]:
    return (label, ok, detail, fix)


def doctor() -> None:
    """Run a full environment health check."""
    banner("doctor")

    checks: list[tuple[str, bool, str, str]] = []

    # Python version
    v = sys.version_info
    py_ok = v >= (3, 11)
    checks.append(_check("Python >= 3.11", py_ok, f"{v.major}.{v.minor}.{v.micro}",
                          "upgrade Python" if not py_ok else ""))

    # ffmpeg
    try:
        from ..core.binaries import get_ffmpeg
        ff = get_ffmpeg()
        r = subprocess.run([ff, "-version"], capture_output=True, text=True, timeout=5)
        ff_ok = r.returncode == 0
        version_line = r.stdout.splitlines()[0] if r.stdout else ""
        checks.append(_check("ffmpeg", ff_ok, version_line[:60] if ff_ok else "not found",
                              "" if ff_ok else "install ffmpeg and add to PATH"))
    except Exception as e:
        checks.append(_check("ffmpeg", False, str(e)[:60], "install ffmpeg and add to PATH"))

    # ffprobe
    try:
        from ..core.binaries import get_ffprobe
        fp = get_ffprobe()
        r = subprocess.run([fp, "-version"], capture_output=True, text=True, timeout=5)
        fp_ok = r.returncode == 0
        version_line = r.stdout.splitlines()[0] if r.stdout else ""
        checks.append(_check("ffprobe", fp_ok, version_line[:60] if fp_ok else "not found",
                              "" if fp_ok else "install ffprobe and add to PATH"))
    except Exception as e:
        checks.append(_check("ffprobe", False, str(e)[:60], "install ffprobe and add to PATH"))

    # Write access to temp dir
    try:
        with tempfile.TemporaryDirectory() as td:
            test = Path(td) / "amverge_write_test.txt"
            test.write_text("ok")
            write_ok = test.exists()
        checks.append(_check("Temp dir writable", write_ok, tempfile.gettempdir()))
    except Exception as e:
        checks.append(_check("Temp dir writable", False, str(e)[:60], "check disk space / permissions"))

    # Base deps
    for pkg, imp in [("av", "av"), ("numpy", "numpy"), ("pillow", "PIL"), ("rich", "rich"), ("typer", "typer")]:
        try:
            mod = __import__(imp)
            ver = getattr(mod, "__version__", "?")
            checks.append(_check(pkg, True, f"v{ver}"))
        except ImportError:
            checks.append(_check(pkg, False, "not installed", f"pip install {pkg}"))

    # ML deps
    try:
        import torch
        checks.append(_check("torch [ml]", True, f"v{torch.__version__}  CUDA={'yes' if torch.cuda.is_available() else 'no'}"))
    except ImportError:
        checks.append(_check("torch [ml]", False, "not installed", "pip install amverge[ml]"))

    try:
        import transnetv2_pytorch as tv2
        checks.append(_check("transnetv2-pytorch [ml]", True, getattr(tv2, "__version__", "installed")))
    except ImportError:
        checks.append(_check("transnetv2-pytorch [ml]", False, "not installed", "pip install amverge[ml]"))

    try:
        import tqdm
        checks.append(_check("tqdm [ml]", True, f"v{tqdm.__version__}"))
    except ImportError:
        checks.append(_check("tqdm [ml]", False, "not installed", "pip install amverge[ml]"))

    # Optional deps
    try:
        import cv2
        checks.append(_check("opencv [edge]", True, f"v{cv2.__version__}"))
    except ImportError:
        checks.append(_check("opencv [edge]", False, "not installed", "pip install amverge[edge]"))

    try:
        import pypresence
        checks.append(_check("pypresence [discord]", True, getattr(pypresence, "__version__", "installed")))
    except ImportError:
        checks.append(_check("pypresence [discord]", False, "not installed", "pip install amverge[discord]"))

    # nelux
    try:
        from ..core.nelux_runtime import _get_nelux_video_reader
        _get_nelux_video_reader()
        checks.append(_check("nelux", True, "available"))
    except ImportError as e:
        if "Failed to import nelux" in str(e):
            checks.append(_check("nelux", False, "DLLs not found", "set AMVERGE_FFMPEG_BIN env var"))
        else:
            checks.append(_check("nelux", False, "not installed", "optional - Windows native decoder"))
    except Exception:
        checks.append(_check("nelux", False, "not installed", "optional - Windows native decoder"))

    passed = sum(1 for _, ok, _, _ in checks if ok)
    total = len(checks)

    t = make_table(
        ("",        "muted",  {"width": 26, "no_wrap": True}),
        ("",        "label",  {"width": 3,  "no_wrap": True}),
        ("",        "muted",  {}),
        title=f"Health Check  {passed}/{total} passed",
    )

    for label, ok, detail, fix in checks:
        status = "[accent]pass[/]" if ok else "[error]FAIL[/]"
        note = detail if ok else (f"[error]{detail}[/]" + (f"  [muted]{fix}[/]" if fix else ""))
        t.add_row(label, status, note)

    console.print(t)

    if passed == total:
        console.print("[accent]All checks passed.[/]\n")
    else:
        failed = total - passed
        console.print(f"[error]{failed} check(s) failed.[/]  See fix hints above.\n")
