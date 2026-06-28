"""Scene export with codec profiles, audio selection, and hardware acceleration.

Usage:
    >>> from amverge import SceneExporter
    >>> exporter = SceneExporter(codec="h264_main", audio="aac_320", container="mp4")
    >>> exporter.export(scenes, output_dir="./export")
    >>> exporter.export_one(scene_dict, "clip.mp4")
    >>> exporter.merge(scenes, "merged.mkv")
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


class SceneExporter:
    """Configurable exporter for scene clips.

    Args:
        codec: Codec profile key (e.g. ``"h264_main"``, ``"copy"``).
            Aliases ``"h264"``, ``"hevc"``, ``"h265"`` are resolved.
        audio: Audio codec key (e.g. ``"aac_320"``, ``"copy"``, ``"mp3"``).
        container: Output container ``"mp4"``, ``"mkv"``, or ``"mov"``.
        hardware: ``"auto"``, ``"gpu"``, or ``"cpu"``. GPU uses NVENC
            when available. Ignored for ``"copy"`` and ProRes codecs.
        ffmpeg: Path to ffmpeg binary. Auto-detected if None.

    Example:
        >>> exporter = SceneExporter(codec="h264_high", audio="aac_320", hardware="gpu")
        >>> exporter.export(scenes, "./out")
        >>> exporter.merge(scenes, "merged.mp4")
    """

    def __init__(
        self,
        codec: str = "copy",
        *,
        audio: str = "copy",
        container: str = "mp4",
        hardware: str = "auto",
        ffmpeg: str | None = None,
    ) -> None:
        from .codec_utils import (
            CODEC_ALIASES, CODEC_PROFILES, AUDIO_FFMPEG,
            PRORES_CODECS, resolve_gpu,
        )
        self.codec = CODEC_ALIASES.get(codec, codec)
        self.audio = audio
        self.container = container
        self.hardware = hardware

        self._use_gpu = resolve_gpu(hardware, self.codec)
        self._audio_args = AUDIO_FFMPEG.get(audio, ["-c:a", "copy"])
        self._profile = CODEC_PROFILES.get(self.codec, {})
        self._is_prores = self.codec in PRORES_CODECS

        if self._is_prores and container != "mov":
            raise ValueError(f"ProRes codec '{self.codec}' requires container='mov'")

        self._ffmpeg = ffmpeg
        if self._ffmpeg is None:
            from .binaries import get_ffmpeg
            self._ffmpeg = get_ffmpeg()

    def _run(self, cmd: list[str], timeout: int = 300) -> None:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, creationflags=CREATE_NO_WINDOW)
        if p.returncode != 0:
            tail = (p.stderr or "")[-600:]
            raise RuntimeError(f"ffmpeg failed (exit {p.returncode}): {tail}")

    def _build_video_args(self) -> list[str]:
        if self.codec == "copy":
            return ["-c:v", "copy"]
        encoder = self._profile.get("gpu") if self._use_gpu and self._profile.get("gpu") else self._profile.get("cpu", "libx264")
        args = [str(encoder)]
        extra = str(self._profile.get("args", "")).strip()
        if extra:
            args += extra.split()
        return ["-c:v"] + args

    def export(
        self,
        scenes: list[dict[str, Any]],
        output_dir: str | Path,
        *,
        select: list[int] | None = None,
    ) -> list[str]:
        """Export selected scenes to individual files.

        Args:
            scenes: List of scene dicts with ``"scene_index"`` and ``"path"`` keys.
            output_dir: Output directory.
            select: Optional list of scene indices to export (all if None).

        Returns:
            List of output file paths.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        to_export = [
            s for s in scenes
            if select is None or (s.get("scene_index") or s.get("index")) in select
        ]

        results: list[str] = []

        def _one(scene: dict) -> str:
            idx = scene.get("scene_index", scene.get("index", 0))
            dst = str(out_dir / f"scene_{idx:04d}.{self.container}")
            cmd = [self._ffmpeg, "-y"]
            cmd += ["-i", scene["path"]]
            cmd += self._build_video_args()
            cmd += self._audio_args
            cmd.append(dst)
            self._run(cmd)
            return dst

        max_w = min(4, len(to_export), (os.cpu_count() or 4))
        with ThreadPoolExecutor(max_workers=max_w + 1) as executor:
            futures = {executor.submit(_one, s): s for s in to_export}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception:
                    pass

        return sorted(results)

    def export_one(
        self,
        scene: dict[str, Any],
        output: str | Path,
    ) -> str:
        """Export a single scene to a specific file.

        Args:
            scene: Scene dict with ``"path"`` key.
            output: Output file path.

        Returns:
            Path to the output file.
        """
        dst = str(Path(output).resolve())
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        cmd = [self._ffmpeg, "-y"]
        cmd += ["-i", scene["path"]]
        cmd += self._build_video_args()
        cmd += self._audio_args
        cmd.append(dst)
        self._run(cmd)
        return dst

    def merge(
        self,
        scenes: list[dict[str, Any]],
        output: str | Path,
        *,
        select: list[int] | None = None,
    ) -> str:
        """Merge selected scenes into a single file using ffmpeg concat.

        Args:
            scenes: List of scene dicts with ``"path"`` keys.
            output: Output file path.
            select: Optional list of scene indices to merge (all if None).

        Returns:
            Path to the merged output file.
        """
        dst = str(Path(output).resolve())
        Path(dst).parent.mkdir(parents=True, exist_ok=True)

        to_merge = [
            s for s in scenes
            if select is None or (s.get("scene_index") or s.get("index")) in select
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            concat_file = f.name
            for s in to_merge:
                path = s["path"].replace("\\", "/")
                f.write(f"file '{path}'\n")

        try:
            cmd = [self._ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", concat_file]
            cmd += self._build_video_args()
            cmd += self._audio_args
            cmd.append(dst)
            self._run(cmd)
        finally:
            os.unlink(concat_file)

        return dst
