"""Vendor-neutral GPU detection for routing work to CUDA or Vulkan backends.

AMVerge runs AI models through two families of backend: PyTorch CUDA (NVIDIA
only) and Vulkan (vendor-neutral, used by Anime4K via libplacebo and by the
Flowframes ncnn engines). This module answers "what GPU is in this machine"
without assuming the answer is NVIDIA.

Usage:
    >>> from amverge.core.infra.device import detect_gpu, gpu_vendor
    >>> gpu = detect_gpu()
    >>> print(gpu.vendor, gpu.name, gpu.vram_gb)
    amd Radeon RX 7900 XTX 24.0
"""

from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

VENDOR_NVIDIA = "nvidia"
VENDOR_AMD = "amd"
VENDOR_INTEL = "intel"
VENDOR_NONE = "none"

_MARKERS = (
    (VENDOR_NVIDIA, (r"nvidia", r"geforce", r"\brtx\b", r"\bgtx\b", r"quadro", r"tesla")),
    (VENDOR_AMD, (r"\bamd\b", r"radeon", r"\brx\s*\d", r"\bvega\b", r"firepro", r"\bnavi\b")),
    (VENDOR_INTEL, (r"\bintel\b", r"\barc\b", r"\buhd\b", r"\biris\b")),
)

_PS_ADAPTERS = (
    "Get-CimInstance Win32_VideoController | "
    "ForEach-Object { $_.Name + '|' + $_.AdapterRAM + '|' + $_.DriverVersion "
    "+ '|' + $_.ConfigManagerErrorCode }"
)

_REG_QUERY = (
    "Get-ItemProperty -Path 'HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Class\\"
    "{4d36e968-e325-11ce-bfc1-08002be10318}\\*' -ErrorAction SilentlyContinue | "
    "Where-Object { $_.'HardwareInformation.qwMemorySize' } | "
    "ForEach-Object { $_.DriverDesc + '|' + $_.'HardwareInformation.qwMemorySize' }"
)

_cache: "GpuDevice | None" = None


@dataclass(frozen=True)
class GpuDevice:
    """A detected GPU.

    Attributes:
        vendor: one of ``nvidia``, ``amd``, ``intel``, ``none``.
        name: adapter name as reported by the driver, or ``None``.
        vram_gb: total video memory in GiB, ``0.0`` when unknown.
        driver: driver version string, or ``None``.
        torch_cuda: ``True`` when PyTorch can run on this GPU via CUDA.
    """
    vendor: str = VENDOR_NONE
    name: str | None = None
    vram_gb: float = 0.0
    driver: str | None = None
    torch_cuda: bool = False

    @property
    def is_nvidia(self) -> bool:
        return self.vendor == VENDOR_NVIDIA

    @property
    def is_amd(self) -> bool:
        return self.vendor == VENDOR_AMD

    @property
    def available(self) -> bool:
        return self.vendor != VENDOR_NONE

    def to_dict(self) -> dict:
        return {
            "vendor": self.vendor,
            "name": self.name,
            "vram_gb": self.vram_gb,
            "driver": self.driver,
            "torch_cuda": self.torch_cuda,
        }


def classify_vendor(name: str | None) -> str:
    """Map an adapter name to a vendor constant."""
    if not name:
        return VENDOR_NONE
    lowered = name.lower()
    for vendor, patterns in _MARKERS:
        for pattern in patterns:
            if re.search(pattern, lowered):
                return vendor
    return VENDOR_NONE


def _run_ps(command: str) -> str:
    """Run a PowerShell snippet and return its stdout.

    The command is passed as an argv list, never through a shell. Quoting a
    registry path through a shell mangles the GUID braces and backslashes and
    silently yields nothing. PowerShell also reports a non-zero exit for a
    trailing non-terminating error while still emitting valid output, so the
    exit code is not consulted.
    """
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True, text=True, timeout=20,
            creationflags=CREATE_NO_WINDOW,
        )
    except Exception:
        return ""
    return (proc.stdout or "").strip()


def _registry_vram() -> dict:
    """Map adapter name to VRAM bytes from the driver registry key.

    Win32_VideoController.AdapterRAM is a uint32 and saturates at 4 GiB, which
    misreports every modern card. The class registry key holds the real size.
    """
    sizes: dict = {}
    for line in _run_ps(_REG_QUERY).splitlines():
        parts = line.strip().split("|")
        if len(parts) < 2 or not parts[1].strip().isdigit():
            continue
        sizes[parts[0].strip().lower()] = int(parts[1].strip())
    return sizes


def _adapter_usable(parts: list[str]) -> bool:
    """Whether a Win32_VideoController row describes a working adapter.

    The class lists adapters that are present but not usable: disabled in
    Device Manager, driver failed to start, resource conflict. They keep their
    name, VRAM and driver version, so a probe that only reads those treats a
    disabled card as the primary GPU and sends the user down a backend that
    cannot run.

    ConfigManagerErrorCode is 0 only when the device is working properly. A row
    without the field is trusted, since an older provider that omits it should
    not black out every GPU.
    """
    if len(parts) < 4:
        return True
    code = parts[3].strip()
    if not code:
        return True
    return code == "0"


def _probe_windows() -> list[GpuDevice]:
    output = _run_ps(_PS_ADAPTERS)
    if not output:
        return []

    reg_sizes = _registry_vram()
    found: list[GpuDevice] = []

    for line in output.splitlines():
        parts = line.strip().split("|")
        if not parts or not parts[0].strip():
            continue
        name = parts[0].strip()
        vendor = classify_vendor(name)
        if vendor == VENDOR_NONE:
            continue
        if not _adapter_usable(parts):
            continue

        vram_bytes = reg_sizes.get(name.lower(), 0)
        if not vram_bytes and len(parts) > 1 and parts[1].strip().isdigit():
            vram_bytes = int(parts[1].strip())

        driver = parts[2].strip() if len(parts) > 2 and parts[2].strip() else None
        found.append(GpuDevice(
            vendor=vendor,
            name=name,
            vram_gb=vram_bytes / (1024 ** 3) if vram_bytes else 0.0,
            driver=driver,
        ))

    return found


def _probe_linux() -> list[GpuDevice]:
    try:
        proc = subprocess.run(
            ["lspci"], capture_output=True, text=True, timeout=10,
        )
    except Exception:
        return []
    if proc.returncode != 0:
        return []

    found: list[GpuDevice] = []
    for line in proc.stdout.splitlines():
        if not re.search(r"vga compatible controller|3d controller|display controller", line, re.I):
            continue
        name = line.split(":", 2)[-1].strip()
        vendor = classify_vendor(name)
        if vendor != VENDOR_NONE:
            found.append(GpuDevice(vendor=vendor, name=name))
    return found


def _probe_torch() -> GpuDevice | None:
    try:
        import torch
    except ImportError:
        return None
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            return None
        props = torch.cuda.get_device_properties(0)
        return GpuDevice(
            vendor=VENDOR_NVIDIA,
            name=torch.cuda.get_device_name(0),
            vram_gb=props.total_memory / (1024 ** 3),
            torch_cuda=True,
        )
    except Exception:
        return None


def detect_gpu(refresh: bool = False) -> GpuDevice:
    """Detect the primary GPU and its vendor.

    Prefers PyTorch when it reports a usable CUDA device, since that is the
    backend the ML paths actually run on. Otherwise falls back to an OS level
    adapter query, which is the only way to see an AMD card when PyTorch is a
    CPU or CUDA build.

    Discrete cards win over integrated ones when both are present.

    Args:
        refresh: bypass the module level cache and probe again.

    Returns:
        A :class:`GpuDevice`. ``vendor`` is ``none`` when nothing was found.
    """
    global _cache
    if _cache is not None and not refresh:
        return _cache

    torch_gpu = _probe_torch()

    if sys.platform == "win32":
        adapters = _probe_windows()
    elif sys.platform.startswith("linux"):
        adapters = _probe_linux()
    else:
        adapters = []

    if torch_gpu is not None:
        match = next(
            (a for a in adapters if a.vendor == VENDOR_NVIDIA and a.vram_gb > 0), None,
        )
        if match is not None and torch_gpu.vram_gb <= 0:
            torch_gpu = GpuDevice(
                vendor=VENDOR_NVIDIA, name=torch_gpu.name, vram_gb=match.vram_gb,
                driver=match.driver, torch_cuda=True,
            )
        elif match is not None:
            torch_gpu = GpuDevice(
                vendor=VENDOR_NVIDIA, name=torch_gpu.name, vram_gb=torch_gpu.vram_gb,
                driver=match.driver, torch_cuda=True,
            )
        _cache = torch_gpu
        return _cache

    for vendor in (VENDOR_NVIDIA, VENDOR_AMD, VENDOR_INTEL):
        match = next((a for a in adapters if a.vendor == vendor), None)
        if match is not None:
            _cache = match
            return _cache

    _cache = GpuDevice()
    return _cache


def gpu_vendor() -> str:
    """Return the primary GPU vendor constant."""
    return detect_gpu().vendor


def get_device_type() -> str:
    """Return the PyTorch device type string the ML paths should use."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def get_torch_device():
    """Return the :class:`torch.device` the ML paths should use."""
    import torch
    return torch.device(get_device_type())


def torch_accelerated() -> bool:
    """Whether the PyTorch ML paths can run on the GPU rather than the CPU."""
    return get_device_type() != "cpu"


def torch_backend_gap() -> str | None:
    """Classify why the PyTorch paths cannot use the GPU, if they cannot.

    Returns ``None`` when PyTorch is already GPU accelerated. Otherwise one of:

    ``no_gpu``
        Nothing was detected. Only the CPU is available.
    ``no_torch_backend``
        A GPU is present that PyTorch has no backend for. This is every AMD and
        Intel card on Windows. The Vulkan paths are the answer.
    ``torch_not_cuda``
        An NVIDIA card is present but PyTorch was installed without CUDA, so
        reinstalling torch fixes it.

    Callers own the wording. This module stays free of CLI vocabulary.
    """
    if torch_accelerated():
        return None
    gpu = detect_gpu()
    if not gpu.available:
        return "no_gpu"
    if gpu.is_nvidia:
        return "torch_not_cuda"
    return "no_torch_backend"
