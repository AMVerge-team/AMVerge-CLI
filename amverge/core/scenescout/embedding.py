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

import contextlib
import importlib.util
import os
import threading
import time
from typing import Callable

import numpy as np

from .types import EMBEDDING_MODEL_VERSION

_MODEL = None
_PROCESSOR = None
_DEVICE: str | None = None
_LOAD_LOCK = threading.RLock()
_PROGRESS_LOCK = threading.Lock()
_PROGRESS_LISTENERS: list[Callable[[int, int], None]] = []
_ACTIVE_LOAD = False
_LAST_PROGRESS: tuple[int, int] = (0, 100)


def subscribe_load_progress(cb: Callable[[int, int], None]) -> Callable[[], None]:
    """Subscribe to ``(done, total)`` updates while the SigLIP 2 model loads/downloads."""
    snapshot: tuple[int, int] | None = None
    with _PROGRESS_LOCK:
        _PROGRESS_LISTENERS.append(cb)
        if _ACTIVE_LOAD:
            snapshot = _LAST_PROGRESS
    if snapshot is not None:
        try:
            cb(*snapshot)
        except Exception:
            pass

    def _unsubscribe() -> None:
        with _PROGRESS_LOCK:
            try:
                _PROGRESS_LISTENERS.remove(cb)
            except ValueError:
                pass

    return _unsubscribe


def _emit_load_progress(pct: int, *, force: bool = False) -> None:
    global _LAST_PROGRESS
    clamped = max(0, min(100, int(pct)))
    with _PROGRESS_LOCK:
        if not force and clamped <= _LAST_PROGRESS[0]:
            return
        _LAST_PROGRESS = (clamped, 100)
        listeners = list(_PROGRESS_LISTENERS)
    for cb in listeners:
        try:
            cb(clamped, 100)
        except Exception:
            pass


class SceneScoutModelUnavailable(RuntimeError):
    """The AI extra is not installed, or no usable device was found."""


def is_available() -> bool:
    """True when the model *could* be loaded. Does not load it.

    Called on the status path, so it must stay cheap and must not raise.
    """
    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("transformers") is not None
    )


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
    from .cache import clear_embedding_cache

    clear_embedding_cache()
    with _LOAD_LOCK:
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
    with _LOAD_LOCK:
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
    from .cache import clear_embedding_cache

    clear_embedding_cache()
    with _LOAD_LOCK:
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


def ensure_loaded(device: str | None = None):
    return _load(device)


def _make_snapshot_tqdm_class(on_bytes: Callable[[float, float | None], None]):
    from tqdm.std import tqdm as base_tqdm

    state_lock = threading.Lock()
    active_bar: list[object | None] = [None]

    class _SnapshotProgressTqdm(base_tqdm):
        def __init__(self, *args, **kwargs):
            kwargs["disable"] = True
            desc = str(kwargs.get("desc") or "")
            unit = kwargs.get("unit")
            super().__init__(*args, **kwargs)
            self._is_byte_bar = unit == "B" and "Downloading bytes" not in desc
            self._tracked_n = float(kwargs.get("initial") or 0)
            if self._is_byte_bar:
                with state_lock:
                    active_bar[0] = self

        def update(self, n=1):
            if n:
                with state_lock:
                    self._tracked_n += float(n)
                    tracked = self._tracked_n
                    total = self.total
                    is_byte = self._is_byte_bar and active_bar[0] is self
                if is_byte:
                    on_bytes(tracked, total)
            return None

        def refresh(self, *args, **kwargs):
            with state_lock:
                tracked = self._tracked_n
                total = self.total
                is_byte = self._is_byte_bar and active_bar[0] is self
            if is_byte:
                on_bytes(tracked, total)
            return None

    return _SnapshotProgressTqdm


def _model_snapshot(
    token: str | None,
    on_download_progress: Callable[[int], None] | None = None,
) -> tuple[str, bool]:
    """Return ``(snapshot_path, downloaded_online)``; two-arg helper when progress is passed."""
    from huggingface_hub import snapshot_download

    kwargs = {
        "repo_id": EMBEDDING_MODEL_VERSION,
        "token": token,
        "allow_patterns": ["*.json", "*.safetensors", "*.model", "*.jinja"],
    }
    try:
        path = snapshot_download(**kwargs, local_files_only=True)
        return (path, False) if on_download_progress is not None else path
    except Exception:
        try:
            online_kwargs = dict(kwargs)
            if on_download_progress is not None:
                def _on_bytes(done_bytes: float, total_bytes: float | None) -> None:
                    if not total_bytes or total_bytes <= 0:
                        return
                    frac = max(0.0, min(1.0, done_bytes / float(total_bytes)))
                    # Reserve 2% -> 85% for the network download; remaining 15% is local load.
                    on_download_progress(2 + int(frac * 83))

                online_kwargs["tqdm_class"] = _make_snapshot_tqdm_class(_on_bytes)
            path = snapshot_download(**online_kwargs, local_files_only=False)
            return (path, True) if on_download_progress is not None else path
        except Exception as exc:
            raise SceneScoutModelUnavailable(
                "Scene Scout's SigLIP 2 model cache is incomplete. Connect to the internet "
                "and retry so the missing files can download. If it keeps failing, delete "
                "the models--google--siglip2-so400m-patch16-naflex folder inside your "
                "Hugging Face cache and retry."
            ) from exc


@contextlib.contextmanager
def _load_heartbeat(start_pct: int, ceiling_pct: int, interval_s: float = 0.25):
    """Gently advance progress up to ``ceiling_pct`` while blocking native C++/PyTorch loads run."""
    stop = threading.Event()

    def _tick() -> None:
        pct = start_pct
        while not stop.wait(interval_s) and pct < ceiling_pct:
            pct += 1
            _emit_load_progress(pct)

    t = threading.Thread(target=_tick, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join(timeout=0.5)


def _load(device: str | None = None):
    """Load and cache the model, processor and resolved device.

    The first call downloads the SigLIP 2 checkpoint through the Hugging Face
    hub if it is not already cached. A token from ``HF_TOKEN`` is forwarded so
    gated checkpoints load.
    """
    from ..infra.ipc import log
    from ..infra.config import configure_app_storage

    configure_app_storage()

    global _MODEL, _PROCESSOR, _DEVICE, _ACTIVE_LOAD, _LAST_PROGRESS

    with _LOAD_LOCK:
        if _MODEL is not None and (device is None or device == _DEVICE):
            if _DEVICE == "cuda" and next(_MODEL.parameters()).device.type == "cpu":
                activate_model()
            return _MODEL, _PROCESSOR, _DEVICE

        with _PROGRESS_LOCK:
            _ACTIVE_LOAD = True
            _LAST_PROGRESS = (0, 100)

        try:
            _emit_load_progress(2, force=True)

            # transformers' lazy __getattr__ is not thread-safe on first import; keep it inside _LOAD_LOCK
            # so daemon._prewarm and _BackgroundEmbedder cannot race on AutoModel/AutoProcessor.
            import warnings
            warnings.filterwarnings("ignore")

            os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
            os.environ["TRANSFORMERS_VERBOSITY"] = "error"

            with _load_heartbeat(2, 14, interval_s=0.35):
                import torch
            _emit_load_progress(15)

            with _load_heartbeat(15, 27, interval_s=0.35):
                from transformers import AutoModel, AutoProcessor
                from transformers.utils import logging as hf_logging
                hf_logging.set_verbosity_error()
                hf_logging.disable_progress_bar()
            _emit_load_progress(28)

            device_str = resolve_device(device)

            if _MODEL is not None and (device is None or device == _DEVICE):
                if device_str == "cuda" and next(_MODEL.parameters()).device.type == "cpu":
                    activate_model()
                _emit_load_progress(100)
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

            snapshot_path, downloaded_online = _model_snapshot(
                token,
                on_download_progress=lambda p: _emit_load_progress(28 + int(((p - 2) / 83.0) * 57)) if p >= 2 else None,
            )
            _emit_load_progress(86 if downloaded_online else 32)
            processor = AutoProcessor.from_pretrained(snapshot_path, local_files_only=True)
            _emit_load_progress(89 if downloaded_online else 38)

            base_kwargs: dict = {
                "pretrained_model_name_or_path": snapshot_path,
                "token": token,
                "dtype": dtype,
                "attn_implementation": attn_impl,
                "low_cpu_mem_usage": True,
            }
            with _load_heartbeat(89 if downloaded_online else 38, 94 if downloaded_online else 88, interval_s=0.2):
                model = AutoModel.from_pretrained(**base_kwargs, local_files_only=True)
            _emit_load_progress(95 if downloaded_online else 90)

            if device_str != "cpu":
                with _load_heartbeat(95 if downloaded_online else 90, 98 if downloaded_online else 96, interval_s=0.2):
                    model = model.to(device_str)
            _emit_load_progress(99)

            model.requires_grad_(False)
            model.eval()

            _MODEL, _PROCESSOR, _DEVICE = model, processor, device_str
            _emit_load_progress(100)
            log(f"[diag] scene scout model ready on {device_str} ({dtype})")
            return model, processor, device_str
        finally:
            with _PROGRESS_LOCK:
                _ACTIVE_LOAD = False
                _LAST_PROGRESS = (0, 100)


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