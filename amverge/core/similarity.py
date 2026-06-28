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
    """Check if two thumbnail images look similar.

    Loads both images, average-pools them to 8x8 blocks, computes cosine
    similarity. Returns True if dissimilarity (1 - cosine) falls below
    ``threshold``. Lower threshold = stricter comparison.

    Args:
        path_a: Path to first thumbnail JPEG.
        path_b: Path to second thumbnail JPEG.
        threshold: Dissimilarity threshold (default 0.10).

    Returns:
        True if the images are considered similar.

    Example:
        >>> check_pair_similar("thumb_0001.jpg", "thumb_0002.jpg")
        False
    """
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
    """Find adjacent scene pairs with visually similar thumbnails.

    Iterates through consecutive scene pairs and calls
    :func:`check_pair_similar` on their thumbnails. Scene dicts must have a
    ``"thumbnail"`` key with the path to a JPEG file, and either
    ``"scene_index"`` or ``"index"`` for scene numbering.

    Args:
        scenes: List of scene dicts with ``thumbnail`` and
            ``scene_index`` (or ``index``) keys.
        threshold: Dissimilarity threshold passed to
            :func:`check_pair_similar`.
        progress_cb: Optional ``callback(done: int, total: int)`` called
            after each pair check.

    Returns:
        List of ``(scene_a, scene_b)`` index pairs that look similar.
        Empty list if no similar pairs found.

    Example:
        >>> scenes = [
        ...     {"scene_index": 0, "thumbnail": "th_0000.jpg"},
        ...     {"scene_index": 1, "thumbnail": "th_0001.jpg"},
        ...]
        >>> pairs = find_similar_pairs(scenes)
        >>> for a, b in pairs:
        ...     print(f"Scenes {a} and {b} look similar")
    """
    pairs: list[tuple[int, int]] = []
    total = max(len(scenes) - 1, 0)

    for i, (sa, sb) in enumerate(zip(scenes, scenes[1:])):
        if check_pair_similar(sa["thumbnail"], sb["thumbnail"], threshold):
            pairs.append((sa.get("scene_index", sa.get("index", i)), sb.get("scene_index", sb.get("index", i + 1))))
        if progress_cb:
            try:
                progress_cb(i + 1, total)
            except Exception:
                pass

    return pairs
