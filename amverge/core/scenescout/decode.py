"""Opening a video for Scene Scout's whole-episode decodes, on the GPU when it can.

Sampling frames and measuring motion both decode every frame of an episode, and
decoding is nearly all of their time. PyAV's hardware decoders produce the same
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


def open_video(path: str | Path):
    """``av.open`` for a full decode, with a GPU decoder attached when one is available."""
    import av

    options = {"err_detect": "ignore_err"}
    device = _hw_device()
    if device is not None:
        try:
            from av.codec.hwaccel import HWAccel

            return av.open(str(path), options=options, hwaccel=HWAccel(device_type=device, allow_software_fallback=True))
        except Exception:  # noqa: BLE001 - a device that will not start leaves the CPU decode
            pass
    return av.open(str(path), options=options)
