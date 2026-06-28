"""Adjacent scene similarity detection.

Usage:
    >>> from amverge import SimilarityChecker
    >>> checker = SimilarityChecker(threshold=0.10)
    >>> checker.are_similar("thumb_a.jpg", "thumb_b.jpg")
    >>> pairs = checker.find_in(scenes)
"""

from __future__ import annotations

from typing import Any, Callable


class SimilarityChecker:
    """Detect visually similar adjacent scenes via cosine similarity.

    Compares thumbnails using average-pooled (8x8) pixel arrays and
    cosine similarity. Pairs with dissimilarity below the threshold
    are flagged as similar.

    Args:
        threshold: Dissimilarity threshold (0.0 - 1.0). Lower = stricter.
            Default 0.10.

    Example:
        >>> checker = SimilarityChecker(threshold=0.05)
        >>> checker.are_similar("thumb_0000.jpg", "thumb_0001.jpg")
        False
        >>>
        >>> scenes = [
        ...     {"scene_index": 0, "thumbnail": "th_0000.jpg"},
        ...     {"scene_index": 1, "thumbnail": "th_0001.jpg"},
        ... ]
        >>> pairs = checker.find_in(scenes)
        >>> for a, b in pairs:
        ...     print(f"Scenes {a} and {b} are similar")
    """

    def __init__(self, threshold: float = 0.10) -> None:
        self.threshold = threshold

    def are_similar(self, thumb_a: str, thumb_b: str) -> bool:
        """Check if two thumbnail images look similar.

        Args:
            thumb_a: Path to first thumbnail JPEG.
            thumb_b: Path to second thumbnail JPEG.

        Returns:
            True if the thumbnails are visually similar.
        """
        from .similarity import check_pair_similar
        return check_pair_similar(thumb_a, thumb_b, self.threshold)

    def find_in(
        self,
        scenes: list[dict[str, Any]],
        *,
        progress_cb: Callable[[int, int], None] | None = None,
    ) -> list[tuple[int, int]]:
        """Find all similar adjacent scene pairs in a scene list.

        Scene dicts must have ``"thumbnail"`` key with JPEG paths and
        either ``"scene_index"`` or ``"index"`` for numbering.

        Args:
            scenes: List of scene dicts with thumbnail paths.
            progress_cb: Optional ``callback(done, total)``.

        Returns:
            List of ``(scene_a_idx, scene_b_idx)`` tuples.
        """
        from .similarity import find_similar_pairs
        return find_similar_pairs(scenes, self.threshold, progress_cb)
