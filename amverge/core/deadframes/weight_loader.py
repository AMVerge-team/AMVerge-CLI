from __future__ import annotations

import os
from typing import Callable, Optional

from ..infra.config import get_amverge_config_dir
from ..infra.model_store import download_with_resume, verify_sha256
from .registry import get_model


def _get_weights_dir():
    return os.path.join(get_amverge_config_dir(), "models", "deadframes")


def get_weight_path(model_key):
    entry = get_model(model_key)
    if entry is None or "file" not in entry:
        raise ValueError(f"Unknown model: {model_key}")
    return os.path.join(_get_weights_dir(), entry["file"])


def is_weight_downloaded(model_key):
    entry = get_model(model_key)
    if entry is None or "file" not in entry:
        raise ValueError(f"Unknown model: {model_key}")
    path = get_weight_path(model_key)
    return os.path.exists(path) and os.path.getsize(path) > 0


def verify_weight_hash(model_key, path=None):
    entry = get_model(model_key)
    expected = entry.get("hash") if entry else None
    return verify_sha256(path or get_weight_path(model_key), expected)


def download_weights(
    model_key: str,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    retries: int = 3,
) -> bool:
    entry = get_model(model_key)
    if entry is None:
        raise ValueError(f"Unknown model: {model_key}")

    if "file" not in entry:
        raise ValueError(f"Model '{model_key}' has no weights file to download.")

    url = entry.get("url")
    if not url:
        raise RuntimeError(f"Model '{model_key}' has no published download URL.")

    dest = get_weight_path(model_key)
    if os.path.exists(dest) and os.path.getsize(dest) > 0 and verify_weight_hash(model_key, dest):
        return True

    label = entry.get("name", model_key)
    try:
        return download_with_resume(
            url,
            dest,
            expected_hash=entry.get("hash"),
            retries=retries,
            progress_cb=progress_cb,
            progress_formatter=lambda pct, _dl, _tot: f"Downloading {label}... {pct}%",
            done_message=f"Downloaded {label}",
        )
    except Exception as e:
        raise RuntimeError(f"Download failed for {model_key}: {e}") from e
