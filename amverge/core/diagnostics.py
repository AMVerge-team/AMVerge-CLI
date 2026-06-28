"""Diagnostic helpers for GPU, dependencies, and environment checks.

Usage:
    >>> from amverge import get_gpu_info, get_versions, check_environment
    >>> info = get_gpu_info()
    >>> result = check_environment()
    >>> print(f"{result.passed}/{result.total} checks passed")
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CheckResult:
    """A single health check result."""
    label: str
    ok: bool
    detail: str = ""
    fix: str = ""

    def __bool__(self) -> bool:
        return self.ok


@dataclass
class EnvironmentCheck:
    """Full environment health check result."""
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for c in self.checks if c.ok)

    @property
    def total(self) -> int:
        return len(self.checks)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def is_healthy(self) -> bool:
        return self.passed == self.total

    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.ok]

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "total": self.total,
            "failed": self.failed,
            "healthy": self.is_healthy,
            "checks": [
                {"label": c.label, "ok": c.ok, "detail": c.detail, "fix": c.fix}
                for c in self.checks
            ],
        }


def get_gpu_info() -> dict:
    """Get GPU and CUDA information.

    Returns a dict with keys:
        torch_version, cuda_available, cuda_version, gpu_count,
        gpu_name, vram_gb, transnetv2_available, opencv_available,
        rpc_available, nelux_available.

    Example:
        >>> info = get_gpu_info()
        >>> if info["cuda_available"]:
        ...     print(f"GPU: {info['gpu_name']} ({info['vram_gb']:.1f} GB)")
    """
    info: dict = {
        "torch_version": None,
        "cuda_available": False,
        "cuda_version": None,
        "gpu_count": 0,
        "gpu_name": None,
        "vram_gb": 0.0,
        "transnetv2_available": False,
        "opencv_available": False,
        "rpc_available": False,
        "nelux_available": False,
    }

    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_count"] = torch.cuda.device_count()
            if torch.cuda.device_count() > 0:
                info["gpu_name"] = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                info["vram_gb"] = props.total_memory / (1024 ** 3)
    except ImportError:
        pass

    try:
        from .scene_detection import TRANSNET_AVAILABLE
        info["transnetv2_available"] = TRANSNET_AVAILABLE
    except ImportError:
        pass

    try:
        import cv2
        info["opencv_available"] = True
    except ImportError:
        pass

    try:
        from .discord_rpc import RPC_AVAILABLE
        info["rpc_available"] = RPC_AVAILABLE
    except ImportError:
        pass

    try:
        from .nelux_runtime import _get_nelux_video_reader
        _get_nelux_video_reader()
        info["nelux_available"] = True
    except (ImportError, Exception):
        pass

    return info


def get_versions() -> dict:
    """Get version info for all dependencies.

    Returns a dict with keys matching package names.

    Example:
        >>> versions = get_versions()
        >>> print(f"amverge {versions['amverge']}")
    """
    from ..__version__ import __version__
    versions: dict = {"amverge": __version__}

    for name in ["av", "numpy", "pillow", "rich", "typer", "tqdm"]:
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "installed")
        except ImportError:
            versions[name] = None

    optional = ["torch", "transnetv2_pytorch", "cv2", "pypresence"]
    for name in optional:
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "installed")
        except ImportError:
            versions[name] = None

    return versions


def check_environment() -> EnvironmentCheck:
    """Run a full environment health check.

    Checks Python version, ffmpeg/ffprobe availability, temp dir write
    access, and all dependency installations. Equivalent to
    ``amverge doctor`` but returns structured data.

    Returns:
        :class:`EnvironmentCheck` with results for each check.

    Example:
        >>> result = check_environment()
        >>> print(f"{result.passed}/{result.total} checks passed")
        >>> if not result.is_healthy:
        ...     for c in result.failures():
        ...         print(f"FAIL: {c.label} - {c.fix}")
    """
    result = EnvironmentCheck()

    v = sys.version_info
    py_ok = v >= (3, 11)
    result.checks.append(CheckResult(
        "Python >= 3.11", py_ok, f"{v.major}.{v.minor}.{v.micro}",
        "upgrade Python" if not py_ok else "",
    ))

    try:
        from .binaries import get_ffmpeg
        ff = get_ffmpeg()
        r = subprocess.run([ff, "-version"], capture_output=True, text=True, timeout=5)
        line = r.stdout.splitlines()[0] if r.stdout else ""
        result.checks.append(CheckResult("ffmpeg", r.returncode == 0, line[:80],
            "install ffmpeg and add to PATH" if r.returncode != 0 else ""))
    except Exception as e:
        result.checks.append(CheckResult("ffmpeg", False, str(e)[:80],
            "install ffmpeg and add to PATH"))

    try:
        from .binaries import get_ffprobe
        fp = get_ffprobe()
        r = subprocess.run([fp, "-version"], capture_output=True, text=True, timeout=5)
        line = r.stdout.splitlines()[0] if r.stdout else ""
        result.checks.append(CheckResult("ffprobe", r.returncode == 0, line[:80],
            "install ffprobe and add to PATH" if r.returncode != 0 else ""))
    except Exception as e:
        result.checks.append(CheckResult("ffprobe", False, str(e)[:80],
            "install ffprobe and add to PATH"))

    try:
        with tempfile.TemporaryDirectory() as td:
            test = Path(td) / "amverge_write_test.txt"
            test.write_text("ok")
            ok = test.exists()
        result.checks.append(CheckResult("Temp dir writable", ok, tempfile.gettempdir()))
    except Exception as e:
        result.checks.append(CheckResult("Temp dir writable", False, str(e)[:80],
            "check disk space / permissions"))

    for pkg, imp in [("av", "av"), ("numpy", "numpy"), ("pillow", "PIL"), ("rich", "rich"), ("typer", "typer")]:
        try:
            mod = __import__(imp)
            ver = getattr(mod, "__version__", "installed")
            result.checks.append(CheckResult(pkg, True, f"v{ver}"))
        except ImportError:
            result.checks.append(CheckResult(pkg, False, "not installed", f"pip install {pkg}"))

    for pkg, imp, extra in [
        ("torch", "torch", "[ml]"),
        ("transnetv2-pytorch", "transnetv2_pytorch", "[ml]"),
        ("tqdm", "tqdm", "[ml]"),
        ("opencv", "cv2", "[edge]"),
        ("pypresence", "pypresence", "[discord]"),
    ]:
        try:
            mod = __import__(imp)
            ver = getattr(mod, "__version__", "installed")
            cu = ""
            if imp == "torch":
                cu = f"  CUDA={'yes' if mod.cuda.is_available() else 'no'}"
            result.checks.append(CheckResult(f"{pkg} {extra}", True, f"v{ver}{cu}"))
        except ImportError:
            result.checks.append(CheckResult(f"{pkg} {extra}", False, "not installed",
                f"pip install {extra}"))

    try:
        from .nelux_runtime import _get_nelux_video_reader
        _get_nelux_video_reader()
        result.checks.append(CheckResult("nelux", True, "available"))
    except ImportError as e:
        if "Failed to import nelux" in str(e):
            result.checks.append(CheckResult("nelux", False, "DLLs not found",
                "set AMVERGE_FFMPEG_BIN env var"))
        else:
            result.checks.append(CheckResult("nelux", False, "not installed",
                "optional - Windows native decoder"))
    except Exception:
        result.checks.append(CheckResult("nelux", False, "not installed",
            "optional - Windows native decoder"))

    return result
