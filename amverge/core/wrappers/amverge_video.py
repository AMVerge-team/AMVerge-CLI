"""High-level video object wrapping PyAV, ffprobe, and cutting.

Usage:
    >>> from amverge import AmvergeVideo
    >>> video = AmvergeVideo("episode.mp4")
    >>> print(video.duration, video.fps, video.width, video.height)
    >>> print(video.codec, video.is_hevc)
    >>> video.extract_thumbnail(5.0, "thumb.jpg")
    >>> video.cut_scene(0.0, 10.0, "clip.mp4")
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from functools import cached_property

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


class AmvergeVideo:
    """Video file with lazy-loaded metadata and cutting capability.

    Properties are cached after first access. No frames are decoded
    until you call :meth:`decode_frames` or :meth:`extract_thumbnail`.

    Args:
        path: Path to the video file. Raises FileNotFoundError if missing.

    Example:
        >>> video = AmvergeVideo("episode.mp4")
        >>> print(f"{video.width}x{video.height} {video.fps}fps {video.duration:.1f}s")
        >>> print(f"HEVC: {video.is_hevc}")
        >>> for ts in video.keyframes[:5]:
        ...     print(f"keyframe at {ts:.2f}s")
        >>> video.cut_scene(0.0, 5.0, "scene.mp4")
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path).resolve()
        if not self._path.exists():
            raise FileNotFoundError(f"Video not found: {self._path}")

    # -- basic accessors -------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    @property
    def stem(self) -> str:
        return self._path.stem

    @property
    def name(self) -> str:
        return self._path.name

    def __repr__(self) -> str:
        return f"AmvergeVideo({self.name!r})"

    # -- metadata (PyAV) ------------------------------------------------

    @cached_property
    def _metadata(self) -> dict:
        from ..video.video import get_video_info
        return get_video_info(str(self._path))

    @property
    def duration(self) -> float:
        return self._metadata["duration"]

    @property
    def fps(self) -> float:
        for s in self._metadata["streams"]:
            if s["type"] == "video":
                return s["fps"]
        return 0.0

    @property
    def width(self) -> int:
        for s in self._metadata["streams"]:
            if s["type"] == "video":
                return s["width"]
        return 0

    @property
    def height(self) -> int:
        for s in self._metadata["streams"]:
            if s["type"] == "video":
                return s["height"]
        return 0

    @property
    def codec(self) -> str:
        for s in self._metadata["streams"]:
            if s["type"] == "video":
                return s["codec"]
        return "unknown"

    @property
    def total_frames(self) -> int:
        return int(self.fps * self.duration)

    @property
    def audio_streams(self) -> list[dict]:
        return [s for s in self._metadata["streams"] if s["type"] == "audio"]

    # -- HEVC detection --------------------------------------------------

    @cached_property
    def is_hevc(self) -> bool:
        from ..codec.codec_utils import check_if_hevc
        return check_if_hevc(str(self._path))

    # -- keyframes -------------------------------------------------------

    @cached_property
    def keyframes(self) -> list[float]:
        from ..keyframes.keyframe_align import get_keyframe_timestamps_pyav
        return get_keyframe_timestamps_pyav(str(self._path))

    # -- ffmpeg helper ----------------------------------------------------

    @cached_property
    def _ffmpeg(self) -> str:
        from ..infra.binaries import get_ffmpeg
        return get_ffmpeg()

    def _run_ffmpeg(self, cmd: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, creationflags=CREATE_NO_WINDOW)
        if p.returncode != 0:
            tail = (p.stderr or "")[-600:]
            raise RuntimeError(f"ffmpeg failed (exit {p.returncode}): {tail}")
        return p

    # -- cutting ---------------------------------------------------------

    def cut_scene(
        self,
        start: float,
        end: float,
        output: str | Path,
        *,
        use_cuda: bool | None = None,
    ) -> tuple[str, str]:
        """Cut a time range from the video. Returns ``(path, mode)``.

        Mode is one of ``"copy"``, ``"snapped_copy"``, ``"smartcut"``,
        or ``"reencode"``.

        Args:
            start: Start time in seconds.
            end: End time in seconds.
            output: Output file path.
            use_cuda: Enable GPU encode for re-encode fallback.
                Auto-detected from CUDA availability if None.

        Example:
            >>> path, mode = video.cut_scene(0.0, 5.0, "scene.mp4")
        """
        from ..cutting.smart_cut import cut_scene as _cut
        if use_cuda is None:
            try:
                import torch
                use_cuda = torch.cuda.is_available()
            except ImportError:
                use_cuda = False

        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        result_path, mode = _cut(
            self._path, start, end, 0, out_path.parent,
            self.keyframes, use_cuda, self.is_hevc,
        )
        if Path(result_path) != out_path:
            import shutil
            shutil.move(result_path, out_path)
        return str(out_path), mode

    def copy_segment(
        self,
        start: float,
        end: float,
        output: str | Path,
        *,
        audio: str = "copy",
    ) -> str:
        """Lossless copy of a time range using ffmpeg stream copy.

        Args:
            start: Start time in seconds.
            end: End time in seconds.
            output: Output file path.
            audio: Audio codec (default ``"copy"``, otherwise re-encoded).

        Returns:
            Path to the output file.
        """
        from ..codec.codec_utils import AUDIO_FFMPEG
        out = str(Path(output).resolve())
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self._ffmpeg, "-y",
            "-ss", f"{start:.3f}",
            "-i", str(self._path),
            "-t", f"{end - start:.3f}",
            "-map", "0:v:0", "-map", "0:a?",
            "-c:v", "copy",
        ]
        cmd += AUDIO_FFMPEG.get(audio, ["-c:a", "copy"])
        cmd.append(out)
        self._run_ffmpeg(cmd)
        return out

    def extract_segment(
        self,
        start: float,
        end: float,
        output: str | Path,
        *,
        codec: str = "libx264",
        audio: str = "aac",
        use_cuda: bool = False,
    ) -> str:
        """Re-encode a time range to a new file.

        Args:
            start: Start time in seconds.
            end: End time in seconds.
            output: Output file path.
            codec: Video encoder name (e.g. ``"libx264"``, ``"h264_nvenc"``).
            audio: Audio codec key from ``AUDIO_FFMPEG``.
            use_cuda: Use GPU encoder.

        Returns:
            Path to the output file.
        """
        from ..codec.codec_utils import AUDIO_FFMPEG
        out = str(Path(output).resolve())
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self._ffmpeg, "-y",
            "-ss", f"{start:.3f}",
            "-i", str(self._path),
            "-t", f"{end - start:.3f}",
            "-map", "0:v:0", "-map", "0:a?",
            "-c:v", codec,
        ]
        cmd += AUDIO_FFMPEG.get(audio, ["-c:a", "aac"])
        cmd.append(out)
        self._run_ffmpeg(cmd)
        return out

    # -- thumbnails ------------------------------------------------------

    def extract_thumbnail(
        self,
        at_time: float = 0.0,
        output: str | Path = "",
    ) -> str:
        """Extract a frame as JPEG thumbnail at the given time.

        Args:
            at_time: Timestamp in seconds (snaps to nearest keyframe).
            output: Output path or "" for auto-named file.

        Returns:
            Path to the thumbnail JPEG.
        """
        out = Path(output) if output else self._path.parent / f"{self.stem}_thumb.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self._ffmpeg, "-y",
            "-ss", f"{at_time:.3f}",
            "-i", str(self._path),
            "-vframes", "1",
            "-q:v", "2",
            str(out),
        ]
        self._run_ffmpeg(cmd)
        return str(out)

    def extract_thumbnails_at(
        self,
        timestamps: list[float],
        output_dir: str | Path,
        *,
        prefix: str = "thumb",
        workers: int = 4,
    ) -> list[str]:
        """Extract thumbnails at multiple timestamps in parallel.

        Args:
            timestamps: List of seconds to extract frames at.
            output_dir: Directory for output JPEGs.
            prefix: Filename prefix.
            workers: Number of parallel ffmpeg calls.

        Returns:
            List of output file paths.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        def _one(i: int, ts: float) -> str:
            out = out_dir / f"{prefix}_{i:04d}.jpg"
            cmd = [
                self._ffmpeg, "-y",
                "-ss", f"{ts:.3f}",
                "-i", str(self._path),
                "-vframes", "1",
                "-q:v", "2",
                str(out),
            ]
            self._run_ffmpeg(cmd)
            return str(out)

        results: list[str] = []
        max_w = min(int(workers), len(timestamps), (os.cpu_count() or 4))
        with ThreadPoolExecutor(max_workers=max_w) as executor:
            futures = {executor.submit(_one, i, ts): i for i, ts in enumerate(timestamps)}
            for future in as_completed(futures):
                results.append(future.result())
        return sorted(results)

    # -- frame decoding --------------------------------------------------

    def decode_frames(self) -> "np.ndarray":
        """Decode all frames for TransNetV2 processing.

        Returns:
            ndarray of shape ``(N, 27, 48, 3)`` with dtype ``uint8``.
        """
        try:
            from ..detection.ai_scene_detection import decode_video_frames_nelux
            return decode_video_frames_nelux(str(self._path))
        except ImportError:
            import numpy as np
            from ..transnet.transnet_constants import FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS, FRAME_BYTES
            cmd = [
                self._ffmpeg, "-y",
                "-i", str(self._path),
                "-pix_fmt", "rgb24",
                "-vf", "scale=48:27",
                "-f", "rawvideo",
                "pipe:1",
            ]
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            if p.stdout is None:
                raise RuntimeError("Failed to create ffmpeg stdout pipe")
            total = max(1, self.total_frames)
            frames = np.empty((total, FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS), dtype=np.uint8)
            for i in range(total):
                raw = p.stdout.read(FRAME_BYTES)
                if len(raw) == 0:
                    return frames[:i]
                frames[i] = np.frombuffer(raw, dtype=np.uint8).reshape(
                    FRAME_HEIGHT, FRAME_WIDTH, FRAME_CHANNELS
                )
            return frames

    # -- detection -------------------------------------------------------

    def detect_scenes(
        self,
        method: str = "keyframe",
        output_dir: str | Path | None = None,
        **kwargs,
    ) -> "DetectResult":
        """Run scene detection on this video.

        Args:
            method: ``"keyframe"``, ``"edge"``, or ``"transnetv2"``.
            output_dir: Output directory (defaults to ``<stem>_scenes/``).
            **kwargs: Passed to :func:`amverge.detect_scenes`.

        Returns:
            :class:`~amverge.DetectResult` with scenes and similar pairs.
        """
        from ...pipeline import detect_scenes as _detect
        out = str(Path(output_dir).resolve()) if output_dir else None
        return _detect(str(self._path), output_dir=out, method=method, **kwargs)

    def to_dict(self) -> dict:
        """Serialize video metadata to a plain dict."""
        return {
            "path": str(self._path),
            "stem": self.stem,
            "name": self.name,
            "duration": self.duration,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "codec": self.codec,
            "is_hevc": self.is_hevc,
            "total_frames": self.total_frames,
            "keyframe_count": len(self.keyframes),
        }
