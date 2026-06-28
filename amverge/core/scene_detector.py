"""Scene detection with configurable methods and progress.

Usage:
    >>> from amverge import SceneDetector
    >>> detector = SceneDetector(method="transnetv2")
    >>> result = detector.detect("episode.mp4")
    >>> for scene in result.scenes[:3]:
    ...     print(scene.index, scene.start, scene.end)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Literal

DetectionMethod = Literal["keyframe", "edge", "transnetv2"]
ProgressCb = Callable[[str, int, str], None]


class SceneDetector:
    """Configurable scene detector wrapping all three detection methods.

    Args:
        method: ``"keyframe"`` (fast, default), ``"edge"`` (needs opencv),
            or ``"transnetv2"`` (best, needs torch).
        min_duration: Merge scenes shorter than this many seconds.
        thumbnails: Generate JPEG thumbnails for each scene.
        similarity: Run adjacent-scene similarity check.
        similarity_threshold: Cosine dissimilarity threshold (lower = stricter).
        thumbnail_workers: Number of parallel thumbnail threads.
        output_dir: Override output directory. Defaults to ``<name>_scenes/``.
        **kwargs: Passed to :func:`amverge.detect_scenes` (e.g. ``edge_threshold``).

    Example:
        >>> detector = SceneDetector(method="keyframe", min_duration=1.0)
        >>> result = detector.detect("episode.mp4")
        >>>
        >>> # Filter out short scenes
        >>> filtered = result.filter(min_duration=2.0)
        >>>
        >>> # Merge similar-looking scenes
        >>> cleaned = result.merge_similar()
        >>>
        >>> # Save to file
        >>> result.to_json("filtered_scenes.json")
    """

    def __init__(
        self,
        method: DetectionMethod = "keyframe",
        *,
        min_duration: float = 0.25,
        thumbnails: bool = True,
        similarity: bool = True,
        similarity_threshold: float = 0.10,
        thumbnail_workers: int = 4,
        output_dir: str | Path | None = None,
        progress: ProgressCb | None = None,
        **kwargs,
    ) -> None:
        self.method = method
        self.min_duration = min_duration
        self.thumbnails = thumbnails
        self.similarity = similarity
        self.similarity_threshold = similarity_threshold
        self.thumbnail_workers = thumbnail_workers
        self.output_dir = str(Path(output_dir).resolve()) if output_dir else None
        self.progress = progress
        self.extra_kwargs = kwargs

    def detect(
        self,
        video: str | Path,
        output_dir: str | Path | None = None,
    ) -> "DetectResult":
        """Run detection on a video file.

        Args:
            video: Path to the source video.
            output_dir: Override output directory for this run.

        Returns:
            :class:`~amverge.DetectResult` with scenes and similar pairs.
        """
        from ..pipeline import detect_scenes

        t0 = time.monotonic()
        result = detect_scenes(
            video_path=str(video),
            output_dir=str(Path(output_dir).resolve()) if output_dir else self.output_dir,
            method=self.method,
            min_duration=self.min_duration,
            thumbnails=self.thumbnails,
            similarity=self.similarity,
            similarity_threshold=self.similarity_threshold,
            thumbnail_workers=self.thumbnail_workers,
            progress=self.progress,
            **self.extra_kwargs,
        )
        elapsed = round(time.monotonic() - t0, 2)

        result.detection_time = elapsed
        result.method = self.method
        result.video_path = str(video)
        return result
