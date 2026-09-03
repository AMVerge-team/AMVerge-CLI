"""Depth model registry and on-disk lookups.

Deliberately free of torch, cv2 and numpy. Listing the models for the app's
model manager only needs this dictionary and a file check, and importing the ML
stack to answer that took seconds every time the settings panel opened.
`depth_map` re-exports these names, so nothing that imported them from there
has to change.
"""

from __future__ import annotations

import os

from ..infra.config import get_amverge_config_dir


# `size_bytes` and `label`/`summary` are metadata for the app's model manager.
# The size is the real download size, so it can be shown before the file exists
# rather than only after; reading it off disk meant the row said nothing until
# the download had already finished.
MODEL_CONFIGS: dict[str, dict] = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
        "url": "https://github.com/moongetsu/AniSmooth-Models/releases/download/depth/depth_anything_v2_vits.pth",
        "file": "depth_anything_v2_vits.pth",
        "size_bytes": 99218434,
        "label": "Small",
        "summary": "Fastest and lightest. Least detail, good for quick previews.",
    },
    "vitb": {
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
        "url": "https://github.com/moongetsu/AniSmooth-Models/releases/download/depth/depth_anything_v2_vitb.pth",
        "file": "depth_anything_v2_vitb.pth",
        "size_bytes": 389961218,
        "label": "Base",
        "summary": "Balanced. Noticeably more accurate than Small for moderate extra time.",
    },
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
        "url": "https://github.com/moongetsu/AniSmooth-Models/releases/download/depth/depth_anything_v2_vitl.pth",
        "file": "depth_anything_v2_vitl.pth",
        "size_bytes": 1341395338,
        "label": "Large",
        "summary": "Most accurate and most detailed. Slowest, and the largest download.",
    },
}


def _get_depth_models_dir() -> str:
    return os.path.join(get_amverge_config_dir(), "models", "depth")


def _get_model_path(encoder: str) -> str:
    config = MODEL_CONFIGS.get(encoder)
    if not config:
        raise ValueError(f"Unknown encoder: {encoder}")
    return os.path.join(_get_depth_models_dir(), config["file"])


def is_model_downloaded(encoder: str) -> bool:
    path = _get_model_path(encoder)
    return os.path.exists(path) and os.path.getsize(path) > 0
