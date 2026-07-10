from __future__ import annotations

import hashlib
import os
import ssl
import urllib.error
import urllib.request
from typing import Callable, Optional

from ..infra.config import get_amverge_config_dir
from .registry import get_model, INTERPOLATION_REGISTRY, get_all_model_keys


def _get_weights_dir():
    return os.path.join(get_amverge_config_dir(), "models", "interpolation")


def get_weight_path(model_key):
    entry = get_model(model_key)
    if entry is None or "file" not in entry:
        raise ValueError(f"Unknown model: {model_key}")
    return os.path.join(_get_weights_dir(), entry["file"])


def is_weight_downloaded(model_key):
    path = get_weight_path(model_key)
    return os.path.exists(path) and os.path.getsize(path) > 0


def verify_weight_hash(model_key, path=None):
    entry = get_model(model_key)
    expected = entry.get("hash") if entry else None
    if not expected:
        return True
    path = path or get_weight_path(model_key)
    if not os.path.exists(path):
        return False
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            sha.update(chunk)
    return sha.hexdigest() == expected


def download_weights(model_key, progress_cb=None, retries=3):
    entry = get_model(model_key)
    if entry is None:
        raise ValueError(f"Unknown model: {model_key}")

    dest = get_weight_path(model_key)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        if verify_weight_hash(model_key, dest):
            return True

    url = entry["url"]
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    temp_path = dest + ".part"
    ctx = ssl._create_unverified_context()

    for attempt in range(retries):
        try:
            existing = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0
            headers = {"User-Agent": "amverge/1.0"}
            if existing > 0 and attempt > 0:
                headers["Range"] = f"bytes={existing}-"
            elif existing > 0:
                os.remove(temp_path)
                existing = 0

            req = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req, timeout=120, context=ctx)
            code = resp.getcode()
            if code not in (200, 206):
                raise urllib.error.HTTPError(url, code, "", None, None)

            total = int(resp.headers.get("Content-Length", 0))
            file_mode = "ab" if code == 206 else "wb"
            downloaded = existing if code == 206 else 0
            if code == 206:
                total = existing + total
            chunk_size = 65536

            with open(temp_path, file_mode) as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_cb and total > 0:
                        pct = min(99, int(downloaded * 100 / total))
                        progress_cb(pct, f"Downloading {entry.get('name', model_key)}... {pct}%")

            if total > 0 and downloaded != total:
                raise ConnectionError(f"Incomplete: {downloaded}/{total} bytes")

            os.rename(temp_path, dest)
            if not verify_weight_hash(model_key, dest):
                raise RuntimeError(f"Hash mismatch for {model_key}")

            if progress_cb:
                progress_cb(100, f"Downloaded {entry.get('name', model_key)}")
            return True

        except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError,
                TimeoutError, OSError) as e:
            if attempt == retries - 1:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except OSError:
                        pass
                raise RuntimeError(f"Download failed for {model_key}: {e}")

    return False


def _extract_rife_state_dict(checkpoint):
    """Unwrap common checkpoint containers and normalize keys to bare
    ``block0.*`` / ``encode.*`` form (strip ``module.`` and ``flownet.``)."""
    from collections import OrderedDict

    sd = checkpoint
    if isinstance(checkpoint, dict):
        for k in checkpoint:
            if isinstance(k, str) and "model_state_dict" in k and k.startswith("ema_"):
                sd = checkpoint[k]
                break
    if not isinstance(sd, dict):
        return OrderedDict()

    out = OrderedDict()
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("flownet."):
            nk = nk[len("flownet."):]
        out[nk] = v
    return out


def load_weights_if_available(model_key, device="cpu"):
    import torch

    entry = get_model(model_key)
    if entry is None:
        raise ValueError(f"Unknown model: {model_key}")

    if not is_weight_downloaded(model_key):
        raise FileNotFoundError(
            f"Weights not found for {model_key}. Run: amverge interpolate --model {model_key} --download"
        )

    path = get_weight_path(model_key)
    checkpoint = torch.load(path, map_location=device, weights_only=True)

    from .rife_arch import RIFEModel, infer_config

    state_dict = _extract_rife_state_dict(checkpoint)
    try:
        cfg = infer_config(state_dict)
    except ValueError as e:
        raise RuntimeError(
            f"Model '{model_key}' uses an unsupported architecture ({e}). "
            f"This build supports RIFE IFNet checkpoints only."
        )

    model = RIFEModel(cfg).to(device)
    model.eval()

    prefixed = {
        (k if k.startswith("flownet.") else f"flownet.{k}"): v
        for k, v in state_dict.items()
    }
    incompatible = model.load_state_dict(prefixed, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint for '{model_key}' does not match its inferred architecture "
            f"(missing={len(incompatible.missing_keys)}, "
            f"unexpected={len(incompatible.unexpected_keys)})."
        )

    return model
