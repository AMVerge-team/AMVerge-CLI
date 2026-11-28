"""Opening and closing a video for Scene Scout's decodes, on the GPU when it can.

Decoding is nearly all of the time spent sampling frames and measuring motion. PyAV's hardware decoders produce the same
pixels as the CPU (checked on 10-bit HEVC: zero difference) about a third
faster, and leave the CPU free. A file the GPU cannot decode (10-bit H.264, an
unusual codec) falls back to the CPU on its own.

``AMVERGE_SCOUT_HWACCEL=0`` turns it off.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

# tried in order; the first one this PyAV build offers is used
_PREFERRED = {
    "win32": ("cuda", "d3d11va"),
    "darwin": ("videotoolbox",),
}
_OTHER = ("cuda", "vaapi")


@lru_cache(maxsize=1)
def _hw_device() -> Optional[str]:
    if os.environ.get("AMVERGE_SCOUT_HWACCEL", "").strip() == "0":
        return None
    try:
        from av.codec.hwaccel import hwdevices_available

        available = set(hwdevices_available())
    except Exception:  # noqa: BLE001 - an older PyAV without the api decodes on the CPU as before
        return None
    return next((d for d in _PREFERRED.get(sys.platform, _OTHER) if d in available), None)


_DISCRETE_HW_BACKENDS = frozenset({"cuda", "d3d11va", "vaapi"})


def _is_high_bit_depth(path: str | Path) -> bool:
    """True when the first video stream is 10-bit or deeper (e.g. yuv420p10le)."""
    import av

    try:
        with av.open(str(path), options={"err_detect": "ignore_err"}) as container:
            stream = next(iter(container.streams.video), None)
            if stream is None:
                return False
            ctx = stream.codec_context
            pix_fmt = (ctx.pix_fmt or "").lower()
            if any(tag in pix_fmt for tag in ("10", "12", "16", "p010", "p016")):
                return True
            bits = getattr(ctx, "bits_per_raw_sample", 0) or 0
            return int(bits) > 8
    except Exception:
        return False


def open_video(path: str | Path):
    """``av.open`` for a decode, using a GPU decoder when it is actually faster."""
    import av

    options = {"err_detect": "ignore_err"}
    device = _hw_device()
    # On discrete PCIe GPUs, 10-bit HW decode forces 6.2 MB/frame p010le readback over PCIe,
    # making sparse CPU decode with thread_type="AUTO" ~1.8x faster and bit-exact.
    if device is not None and not (device in _DISCRETE_HW_BACKENDS and _is_high_bit_depth(path)):
        try:
            from av.codec.hwaccel import HWAccel

            return av.open(str(path), options=options, hwaccel=HWAccel(device_type=device, allow_software_fallback=True))
        except Exception:  # noqa: BLE001 - a device that will not start leaves the CPU decode
            pass
    return av.open(str(path), options=options)


def close_video(container, stream) -> None:
    """Closes a decode, safe to call partway through the video.

    Freeing a frame-threaded decoder that still has packets in flight deadlocks
    (PyAV 17.1, CPU and GPU alike), so a decode that stops early has to drain first.
    """
    import av

    try:
        for _ in stream.codec_context.decode(None):
            pass
    except (av.FFmpegError, ValueError):
        pass
    container.close()
