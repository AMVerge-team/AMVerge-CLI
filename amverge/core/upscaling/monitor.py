import os
import threading
import time
from typing import Optional

_UNSET = object()
_sampler = _UNSET
_sampler_lock = threading.Lock()


def _sample_nvidia_smi():
    try:
        import subprocess
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
            creationflags=0x08000000 if os.name == "nt" else 0,
        )
        if smi.returncode == 0 and smi.stdout.strip():
            parts = [p.strip() for p in smi.stdout.strip().split(",")]
            if len(parts) >= 4:
                return {
                    "gpu_util": float(parts[0]),
                    "vram_used_mb": float(parts[1]),
                    "vram_total_mb": float(parts[2]),
                    "gpu_temp": float(parts[3]),
                }
    except Exception:
        pass
    return None


def _sample_torch_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            return {
                "gpu_util": None,
                "vram_used_mb": (total - free) / (1024 * 1024),
                "vram_total_mb": total / (1024 * 1024),
                "gpu_temp": None,
            }
    except Exception:
        pass
    return None


def _sample_adapter():
    """Report what the OS adapter list knows: total VRAM and nothing live.

    There is no cheap per-vendor utilization source on Windows outside
    nvidia-smi. The GPU performance counters do expose it, but Get-Counter
    takes several seconds per call against a one second sampling interval, so
    it is not usable here. Live fields stay None and the UI renders them as
    unknown rather than stalling the sampler thread.
    """
    try:
        from ..infra.device import detect_gpu
        gpu = detect_gpu()
        if not gpu.available or not gpu.vram_gb:
            return None
        return {
            "gpu_util": None,
            "vram_used_mb": None,
            "vram_total_mb": gpu.vram_gb * 1024,
            "gpu_temp": None,
        }
    except Exception:
        return None


def _select_sampler():
    try:
        from ..infra.device import detect_gpu
        gpu = detect_gpu()
    except Exception:
        gpu = None

    if gpu is None or gpu.is_nvidia:
        for fn in (_sample_nvidia_smi, _sample_torch_cuda):
            if fn() is not None:
                return fn

    if gpu is not None and gpu.available:
        if _sample_adapter() is not None:
            return _sample_adapter

    return None


def sample_gpu():
    """Sample GPU telemetry, or None when nothing can be read.

    The backend is chosen once and cached. Probing on every call meant an AMD
    machine spawned a doomed nvidia-smi process every second for the whole run.
    """
    global _sampler
    if _sampler is _UNSET:
        with _sampler_lock:
            if _sampler is _UNSET:
                _sampler = _select_sampler()
    if _sampler is None:
        return None
    return _sampler()


def sample_cpu():
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            "cpu_percent": psutil.cpu_percent(interval=None),
            "ram_used_gb": mem.used / (1024 ** 3),
            "ram_total_gb": mem.total / (1024 ** 3),
        }
    except ImportError:
        return None


def format_eta(seconds):
    if seconds is None or seconds == float("inf"):
        return "--:--"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


class SystemMonitor:
    def __init__(self, enabled: bool = True, interval: float = 1.0):
        self.enabled = enabled
        self.interval = interval
        self._thread: Optional[threading.Thread] = None
        self._running = False

        self.stats: dict = {
            "pct": 0,
            "msg": "Starting...",
            "elapsed_s": 0.0,
            "eta_s": None,
            "fps": 0.0,
            "gpu_util": None,
            "gpu_temp": None,
            "vram_used_mb": None,
            "vram_total_mb": None,
            "cpu_percent": None,
            "ram_used_gb": None,
            "ram_total_gb": None,
        }
        self._start_time: Optional[float] = None
        self._last_pct = 0

    def start(self):
        self._start_time = time.time()
        self._running = True
        if self.enabled:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self):
        self._running = False
        self.stats["pct"] = 100

    def _loop(self):
        while self._running:
            gpu = sample_gpu()
            cpu = sample_cpu()
            if gpu:
                self.stats.update(gpu)
            if cpu:
                self.stats.update(cpu)
            time.sleep(self.interval)

    def progress_callback(self, pct: int, msg: str):
        elapsed = time.time() - self._start_time if self._start_time else 0
        self.stats["pct"] = pct
        self.stats["msg"] = msg
        self.stats["elapsed_s"] = elapsed

        if pct > 0 and pct < 100 and elapsed > 0:
            self.stats["eta_s"] = (elapsed / pct) * (100 - pct)
        else:
            self.stats["eta_s"] = 0

        if elapsed > 0:
            self._last_pct = pct
