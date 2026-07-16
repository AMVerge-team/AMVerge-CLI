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

VENDOR_NVIDIA = "nvidia"
VENDOR_AMD = "amd"
VENDOR_INTEL = "intel"
VENDOR_NONE = "none"

_MARKERS = (
    (VENDOR_NVIDIA, (r"nvidia", r"geforce", r"\brtx\b", r"\bgtx\b", r"quadro", r"tesla")),
    (VENDOR_AMD, (r"\bamd\b", r"radeon", r"\brx\s*\d", r"\bvega\b", r"firepro", r"\bnavi\b")),
    (VENDOR_INTEL, (r"\bintel\b", r"\barc\b", r"\buhd\b", r"\biris\b")),
)

_DISPLAY_CLASS_KEY = (
    r"SYSTEM\CurrentControlSet\Control\Class"
    r"\{4d36e968-e325-11ce-bfc1-08002be10318}"
)

_PCI_ENUM_KEY = r"SYSTEM\CurrentControlSet\Enum\PCI"

CONFIGFLAG_DISABLED = 0x00000001

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


def _reg_values(key) -> dict:
    import winreg
    out = {}
    for i in range(winreg.QueryInfoKey(key)[1]):
        try:
            name, value, _ = winreg.EnumValue(key, i)
        except OSError:
            continue
        out[name] = value
    return out


def _disabled_pci_ids() -> set:
    """VEN_xxxx&DEV_xxxx ids that are disabled in Device Manager.

    The display class key carries the name and VRAM but not the device state.
    That lives under the PCI enum key, whose ConfigFlags has CONFIGFLAG_DISABLED
    set when the user turns the adapter off. Reading it needs no elevation.
    """
    import winreg

    disabled = set()
    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _PCI_ENUM_KEY)
    except OSError:
        return disabled

    with root:
        i = 0
        while True:
            try:
                dev_id = winreg.EnumKey(root, i)
            except OSError:
                break
            i += 1
            try:
                dev_key = winreg.OpenKey(root, dev_id)
            except OSError:
                continue
            with dev_key:
                j = 0
                while True:
                    try:
                        inst = winreg.EnumKey(dev_key, j)
                    except OSError:
                        break
                    j += 1
                    try:
                        with winreg.OpenKey(dev_key, inst) as inst_key:
                            flags = winreg.QueryValueEx(inst_key, "ConfigFlags")[0]
                    except OSError:
                        continue
                    if isinstance(flags, int) and flags & CONFIGFLAG_DISABLED:
                        disabled.add(_ven_dev(dev_id))
    return disabled


def _ven_dev(device_id: str) -> str:
    """Reduce a PCI id to its VEN_xxxx&DEV_xxxx part, upper case."""
    tail = device_id.upper().split("\\")[-1]
    bits = [b for b in tail.split("&") if b.startswith(("VEN_", "DEV_"))]
    return "&".join(bits)


def _probe_windows() -> list[GpuDevice]:
    """Read the display adapters from the driver registry.

    Win32_VideoController answers the same question through WMI, but
    Get-CimInstance measured around 9 seconds on a laptop with two GPUs, and
    detect_gpu() sits in the startup path of every upscale and interpolate run.
    The registry holds the same fields and answers in about a millisecond.
    """
    import winreg

    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _DISPLAY_CLASS_KEY)
    except OSError:
        return []

    disabled = _disabled_pci_ids()
    found: list[GpuDevice] = []

    with root:
        i = 0
        while True:
            try:
                sub = winreg.EnumKey(root, i)
            except OSError:
                break
            i += 1
            if not sub.isdigit():
                continue
            try:
                key = winreg.OpenKey(root, sub)
            except OSError:
                continue
            with key:
                vals = _reg_values(key)

            name = vals.get("DriverDesc")
            vendor = classify_vendor(name)
            if vendor == VENDOR_NONE:
                continue

            match_id = vals.get("MatchingDeviceId")
            if match_id and _ven_dev(match_id) in disabled:
                continue

            vram = vals.get("HardwareInformation.qwMemorySize") or 0
            if not isinstance(vram, int):
                vram = 0

            found.append(GpuDevice(
                vendor=vendor,
                name=name.strip(),
                vram_gb=vram / (1024 ** 3) if vram else 0.0,
                driver=vals.get("DriverVersion") or None,
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
