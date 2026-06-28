"""Adjacent scene similarity via average-pooled cosine similarity."""
from __future__ import annotations

import numpy as np
from PIL import Image

DISSIM_THRESHOLD = 0.10
POOL_DIM = 8


def _pool(arr: np.ndarray, dim: int) -> np.ndarray:
    h = (arr.shape[0] // dim) * dim
    w = (arr.shape[1] // dim) * dim
    arr = arr[:h, :w]
    if arr.ndim == 3:
        c = arr.shape[2]
        return arr.reshape(h // dim, dim, w // dim, dim, c).mean(axis=(1, 3))
    return arr.reshape(h // dim, dim, w // dim, dim).mean(axis=(1, 3))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a_f = a.flatten().astype(np.float32)
    b_f = b.flatten().astype(np.float32)
    denom = np.linalg.norm(a_f) * np.linalg.norm(b_f)
    return float(np.dot(a_f, b_f) / denom) if denom != 0 else 1.0


def check_pair_similar(path_a: str, path_b: str, threshold: float = DISSIM_THRESHOLD) -> bool:
    try:
        a = np.array(Image.open(path_a).convert("RGB"))
        b = np.array(Image.open(path_b).convert("RGB"))
    except Exception:
        return False

    return (1.0 - _cosine(_pool(a, POOL_DIM), _pool(b, POOL_DIM))) < threshold


def find_similar_pairs(
    scenes: list[dict],
    threshold: float = DISSIM_THRESHOLD,
    progress_cb=None,
) -> list[tuple[int, int]]:
    """Return list of (pos_a, pos_b) pairs that look similar."""
    pairs: list[tuple[int, int]] = []
    total = max(len(scenes) - 1, 0)

    for i, (sa, sb) in enumerate(zip(scenes, scenes[1:])):
        if check_pair_similar(sa["thumbnail"], sb["thumbnail"], threshold):
            pairs.append((sa["scene_index"], sb["scene_index"]))
        if progress_cb:
            try:
                progress_cb(i + 1, total)
            except Exception:
                pass

    return pairs
