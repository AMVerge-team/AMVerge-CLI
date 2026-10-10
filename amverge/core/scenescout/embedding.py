"""SigLIP 2 model loading and embedding.

Two rules the rest of the package depends on:

* **Import torch and transformers lazily, inside functions.** Listing databases,
  reporting status and every ``--help`` must work with no AI extra installed.
  A module-level ``import torch`` makes the whole CLI unusable without it.
* **Embeddings are L2-normalised float32.** :func:`search` compares them with a
  plain dot product and treats the result as cosine similarity. An unnormalised
  vector silently produces nonsense rankings rather than an error.
"""

from __future__ import annotations

import os

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


def _tune_cpu_threads() -> None:
    import platform

    import torch

    cores = None
    try:
        import psutil

        cores = psutil.cpu_count(logical=False)
        if cores is None:
            cores = os.cpu_count()
    except Exception:
        if platform.machine().lower() in ("x86_64", "amd64"):
            cores = max(1, os.cpu_count() // 2)
        else:
            cores = os.cpu_count()

    if cores:
        try:
            torch.set_num_threads(int(cores))
        except Exception:
            pass


def is_model_loaded() -> bool:
    global _MODEL
    return _MODEL is not None


def standby_model() -> bool:
    global _MODEL, _DEVICE
    if _MODEL is None:
        return False
    import torch
    from ..infra.ipc import log
    if _DEVICE == "cuda":
        _MODEL = _MODEL.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log("[diag] scene scout model moved to CPU standby")
        return True
    return False


def activate_model() -> bool:
    global _MODEL, _DEVICE
    if _MODEL is None or _DEVICE != "cuda":
        return False
    import torch
    from ..infra.ipc import log
    if torch.cuda.is_available():
        _MODEL = _MODEL.to("cuda")
        log("[diag] scene scout model resumed on CUDA")
        return True
    return False


def unload_model() -> bool:
    global _MODEL, _PROCESSOR, _DEVICE
    if _MODEL is None and _PROCESSOR is None:
        return False
    import gc
    import torch
    from ..infra.ipc import log
    _MODEL = None
    _PROCESSOR = None
    _DEVICE = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    log("[diag] scene scout model unloaded")
    return True


def _load(device: str | None = None):
    """Load and cache the model, processor and resolved device.

    The first call downloads the SigLIP 2 checkpoint through the Hugging Face
    hub if it is not already cached. A token from ``HF_TOKEN`` is forwarded so
    gated checkpoints load.
    """
    import os
    import warnings
    warnings.filterwarnings("ignore")

    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    os.environ["TRANSFORMERS_VERBOSITY"] = "error"

    import torch
    from transformers import AutoModel, AutoProcessor
    from transformers.utils import logging as hf_logging
    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()

    from ..infra.ipc import log

    global _MODEL, _PROCESSOR, _DEVICE

    device_str = resolve_device(device)

    if _MODEL is not None and (device is None or device == _DEVICE):
        if device_str == "cuda" and next(_MODEL.parameters()).device.type == "cpu":
            activate_model()
        return _MODEL, _PROCESSOR, _DEVICE

    if device_str == "cuda":
        major, _ = torch.cuda.get_device_capability()
        dtype = torch.float16 if major >= 7 else torch.float32
    else:
        dtype = torch.float32

    if device_str == "cpu":
        _tune_cpu_threads()

    log(f"[diag] loading scene scout model on {device_str}...")

    token = os.environ.get("HF_TOKEN") or None
    attn_impl = (
        "sdpa"
        if hasattr(torch.nn.functional, "scaled_dot_product_attention")
        else "eager"
    )

    processor = AutoProcessor.from_pretrained(EMBEDDING_MODEL_VERSION, token=token)

    kwargs: dict = {
        "pretrained_model_name_or_path": EMBEDDING_MODEL_VERSION,
        "token": token,
        "torch_dtype": dtype,
        "attn_implementation": attn_impl,
    }
    model = AutoModel.from_pretrained(**kwargs)

    if device_str != "cpu":
        model = model.to(device_str)

    model.requires_grad_(False)
    model.eval()

    _MODEL, _PROCESSOR, _DEVICE = model, processor, device_str
    log(f"[diag] scene scout model ready on {device_str} ({dtype})")
    return model, processor, device_str


def _normalize(features):
    import torch.nn.functional as F

    return F.normalize(features, p=2, dim=1)


def _extract_features(output):
    import torch

    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        return output.pooler_output
    return output[0]


def _encode_images(images, model, processor, device, max_num_patches=None):
    """Preprocess and encode one batch of images. Rows are NOT normalised yet."""
    import torch

    kwargs = {"images": images, "return_tensors": "pt"}
    if max_num_patches is not None:
        kwargs["max_num_patches"] = max_num_patches

    inputs = processor(**kwargs).to(device)

    if "pixel_values" in inputs and inputs["pixel_values"].is_floating_point():
        inputs["pixel_values"] = inputs["pixel_values"].to(model.dtype)

    with torch.no_grad():
        return _extract_features(model.get_image_features(**inputs))


def embed_text(query: str, *, device: str | None = None, max_patches: int = 256) -> np.ndarray:
    """Embed a search query into the shared image/text space.

    In:  the raw string the user typed, e.g. "a girl standing in the rain".
    Out: ``np.ndarray``, shape ``(D,)``, dtype ``float32``, L2-normalised.

    ``D`` must be the same value :func:`embed_frames` produces. If the two
    disagree, `search.py` skips every row as a version mismatch and searches
    silently return nothing at all, with no error anywhere.
    """
    import torch

    model, processor, device_str = _load(device)

    inputs = processor(
        text=[query.lower()],
        return_tensors="pt",
        padding="max_length",
        max_length=64,
    ).to(device_str)

    with torch.no_grad():
        features = _extract_features(model.get_text_features(**inputs))

    return (
        _normalize(features)
        .cpu()
        .numpy()
        .astype(np.float32)[0]
    )


def embed_image(image_path: str, *, device: str | None = None, max_patches: int = 256) -> np.ndarray:
    """Embed a reference image, for image-to-scene search.

    In:  a path to an image file on disk.
    Out: identical to :func:`embed_text`, shape ``(D,)`` float32 L2-normalised,
         because `search.search()` takes either one without caring which.
    """
    from PIL import Image

    model, processor, device_str = _load(device)

    with Image.open(image_path) as image:
        pil_image = image.convert("RGB")

    features = _encode_images(
        [pil_image], model, processor, device_str, max_num_patches=max_patches
    )

    return (
        _normalize(features)
        .cpu()
        .numpy()
        .astype(np.float32)[0]
    )


def embed_frames(frames: list[np.ndarray], *, device: str | None = None,
                 max_patches: int = 256, batch_size: int = 16,
                 on_progress=None) -> np.ndarray:
    """Embed sampled video frames, one row per frame.

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
    import gc

    from PIL import Image

    model, processor, device_str = _load(device)

    if not frames:
        return np.empty((0, 0), dtype=np.float32)

    rows = []
    for start in range(0, len(frames), batch_size):
        batch = frames[start : start + batch_size]
        images = [Image.fromarray(f) for f in batch]

        features = _encode_images(
            images, model, processor, device_str, max_num_patches=max_patches
        )
        batch_vecs = _normalize(features).cpu().numpy().astype(np.float32)
        rows.append(batch_vecs)

        done = min(start + len(batch), len(frames))
        if on_progress:
            on_progress(done, len(frames))

        if device_str == "cpu":
            del features, batch_vecs
            gc.collect()

    return np.vstack(rows)


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