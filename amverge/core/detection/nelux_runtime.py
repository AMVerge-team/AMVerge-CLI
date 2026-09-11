from __future__ import annotations

"""Loading the Nelux NVDEC decoder.

Nelux wheels are self-contained: each one ships its own FFmpeg DLLs beside the
extension module, so there is nothing to locate and nothing to configure. This
module exists only to import Nelux in the one order it accepts, and to answer
whether it can be used at all.

Nelux is deliberately narrow about what it will load beside:

* wheels are published for Python 3.13 and up only
* each wheel is built against a single PyTorch minor version and refuses to
  import next to any other
* the PyTorch it wants comes from the CUDA 13 index, which supports Turing
  (sm_75) and newer

The app installs it as its own extra for exactly that reason. When any of those
do not hold, the import raises and :func:`nelux_available` answers False, which
sends the caller down the FFmpeg decode path instead.
"""


def _get_nelux_video_reader():
    """Nelux's ``VideoReader`` class.

    Raises:
        ImportError: Nelux is absent, or present but unable to load.
    """
    # nelux refuses to import unless torch is already in sys.modules
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError("Nelux needs PyTorch. Install it with: pip install amverge[ml]") from exc

    # the exception type separates "absent" from "installed but refusing"
    try:
        from nelux import VideoReader
    except ModuleNotFoundError as exc:
        raise ImportError("Nelux is not installed") from exc
    except ImportError as exc:
        # nelux's own message names the PyTorch mismatch when there is one
        raise ImportError(f"Nelux could not be loaded: {exc}") from exc

    return VideoReader


def nelux_available() -> bool:
    """Whether NVDEC decoding through Nelux can be used in this process.

    An import test only. It says nothing about whether a particular file can be
    decoded in hardware - NVDEC turns down 10-bit H.264, for one - so callers
    still need to handle a decode that fails on its own terms.
    """
    try:
        _get_nelux_video_reader()
        return True
    except Exception:
        return False
