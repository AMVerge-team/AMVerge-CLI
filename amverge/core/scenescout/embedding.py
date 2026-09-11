"""SigLIP 2 model loading and embedding.

Feel free to create this however you want, i created kind of a structure to 
make things easy.

Two rules the rest of the package depends on:

* **Import torch and transformers lazily, inside functions.** Listing databases,
  reporting status and every ``--help`` must work with no AI extra installed.
  A module-level ``import torch`` makes the whole CLI unusable without it.
* **Embeddings are L2-normalised float32.** :func:`search` compares them with a
  plain dot product and treats the result as cosine similarity. An unnormalised
  vector silently produces nonsense rankings rather than an error.
"""

from __future__ import annotations

import numpy as np

from .types import EMBEDDING_MODEL_VERSION

_MODEL = None
_PROCESSOR = None
_DEVICE: str | None = None


class SceneScoutModelUnavailable(RuntimeError):
    """The AI extra is not installed, or no usable device was found."""


def is_available() -> bool:
    """True when the model *could* be loaded. Does not load it.

    Called on the status path, so it must stay cheap and must not raise.
    """
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_device(preferred: str | None = None) -> str:
    """Pick a device, honouring an explicit choice when it is actually usable."""
    try:
        import torch
    except ImportError as exc:
        raise SceneScoutModelUnavailable(
            "Scene Scout needs the AI extra. Install it with: pip install amverge[scout]"
        ) from exc

    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load(device: str | None = None, max_patches: int = 256):
    """Load and cache the model.

    TODO(scene-scout): port from src/model_loader.py.

    Expected shape:
        from transformers import AutoModel, AutoProcessor
        processor = AutoProcessor.from_pretrained(EMBEDDING_MODEL_VERSION)
        model = AutoModel.from_pretrained(EMBEDDING_MODEL_VERSION).to(device).eval()

    Note the upstream project sets HF_HUB_ENABLE_HF_TRANSFER for download speed
    and supports several accelerators (dml, xpu, rocm) behind extras. Carry over
    whichever of those AMVerge wants to support; `resolve_device` above only
    covers cuda/mps/cpu.
    """
    global _MODEL, _PROCESSOR, _DEVICE

    if _MODEL is not None and (device is None or device == _DEVICE):
        return _MODEL, _PROCESSOR, _DEVICE

    raise NotImplementedError(
        "Scene Scout model loading is not implemented yet. "
        "Port src/model_loader.py from the scene-scout repo into "
        "amverge/core/scenescout/embedding.py"
    )


def embed_text(query: str, *, device: str | None = None, max_patches: int = 256) -> np.ndarray:
    """Embed a search query into the shared image/text space.

    TODO(scene-scout): port from src/processing.py.

    In:  the raw string the user typed, e.g. "a girl standing in the rain".
    Out: ``np.ndarray``, shape ``(D,)``, dtype ``float32``, L2-normalised.

    ``D`` must be the same value :func:`embed_frames` produces. If the two
    disagree, `search.py` skips every row as a version mismatch and searches
    silently return nothing at all, with no error anywhere.
    """
    raise NotImplementedError(
        "Port the text embedding path from scene-scout's src/processing.py"
    )


def embed_image(image_path: str, *, device: str | None = None, max_patches: int = 256) -> np.ndarray:
    """Embed a reference image, for image-to-scene search.

    TODO(scene-scout): port from src/processing.py.

    In:  a path to an image file on disk.
    Out: identical to :func:`embed_text`, shape ``(D,)`` float32 L2-normalised,
         because `search.search()` takes either one without caring which.
    """
    raise NotImplementedError(
        "Port the image embedding path from scene-scout's src/processing.py"
    )


def embed_frames(frames: list[np.ndarray], *, device: str | None = None,
                 max_patches: int = 256, batch_size: int = 16) -> np.ndarray:
    """Embed sampled video frames, one row per frame.

    TODO(scene-scout): port from src/processing.py.

    In:  a list of ``N`` frames, each ``np.uint8`` of shape ``(H, W, 3)`` in
         **RGB** order. Frames come from :func:`indexing.sample_frames` and are
         not resized first; whatever the processor wants is up to this function.
    Out: ``np.ndarray``, shape ``(N, D)``, dtype ``float32``, every row
         L2-normalised, in the same order as the input.

    Two easy mistakes here, both of which produce a working-looking system that
    returns nonsense rather than an error:

    * **BGR instead of RGB.** OpenCV hands back BGR, PyAV and PIL hand back RGB.
      Embedding BGR frames still yields vectors and still ranks them, just
      against the wrong colours.
    * **Skipping normalisation.** `search.py` scores with a plain dot product and
      treats the result as cosine similarity, which is only true for unit
      vectors.

    ``batch_size`` is what to tune first if VRAM runs out: this runs over every
    scene in the video and is where indexing spends its time.
    """
    raise NotImplementedError(
        "Port the frame embedding path from scene-scout's src/processing.py"
    )


def pack_embedding(vector: np.ndarray) -> bytes:
    """Serialise for the ``embedding`` BLOB column.

    Raw float32 bytes, matching upstream: a database has to stay readable by the
    standalone tool.
    """
    return np.asarray(vector, dtype=np.float32).tobytes()


def unpack_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def model_version() -> str:
    return EMBEDDING_MODEL_VERSION
