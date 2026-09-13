"""`amverge preview-proxy` - make clips playable in a browser-based viewer."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import typer

from ...ui import banner, console, fail


def _emit(payload: dict) -> None:
    """One JSON object per line, flushed.

    A caller feeding a viewer reads these as they arrive and shows each proxy
    the moment it exists. Without the flush, Python holds the whole batch in
    its pipe buffer and the caller sees nothing until the run ends.
    """
    print(json.dumps(payload, separators=(",", ":")), flush=True)


def preview_proxy(
    # not `exists=True`: one missing path should not fail a whole batch
    clips: List[Path] = typer.Argument(..., help="Clips to make playable"),
    height: int = typer.Option(480, "--height", help="Proxy height in pixels"),
    crf: int = typer.Option(30, "--crf", help="Quality, 0-51, lower is better"),
    force: bool = typer.Option(False, "--force", help="Transcode even if already playable"),
    probe_only: bool = typer.Option(False, "--probe", help="Report playability without transcoding"),
    jobs: int = typer.Option(0, "--jobs", help="Clips to transcode at once, 0 picks a default"),
    as_json: bool = typer.Option(False, "--json", help="Output as JSON, one object per line"),
) -> None:
    """Return browser-playable paths for clips, transcoding only what needs it.

    Cutting stream-copies the video, so a clip inherits its source codec. HEVC
    and 10-bit H.264 are both common and neither decodes in a Chromium-based
    viewer: the audio plays and the picture stays black. This produces an 8-bit
    H.264 copy for those, and returns the original untouched for everything else.

    Pass several clips to transcode them in parallel from a single process.
    Each result is printed as it finishes rather than at the end.
    """
    from ...core.preview.proxy import (
        ensure_preview_proxies,
        is_browser_playable,
        probe_video,
    )

    if probe_only:
        for clip in clips:
            if not clip.is_file():
                if as_json:
                    _emit({"original": str(clip), "error": f"No such clip: {clip}"})
                    continue
                fail(f"No such clip: {clip}")
                continue

            info = probe_video(clip)
            playable = is_browser_playable(clip)
            if as_json:
                _emit({
                    "path": str(clip),
                    "original": str(clip),
                    "playable": playable,
                    "codec": info.get("codec_name"),
                    "profile": info.get("profile"),
                    "pixFmt": info.get("pix_fmt"),
                })
                continue
            banner("preview-proxy")
            console.print(f"Clip     : {clip.name}")
            console.print(f"Codec    : {info.get('codec_name')} ({info.get('profile')})")
            console.print(f"Pixel fmt: {info.get('pix_fmt')}")
            console.print(f"Playable : {'yes' if playable else 'no'}")
        return

    if not as_json:
        banner("preview-proxy")

    failed = 0
    results = ensure_preview_proxies(
        clips, height=height, crf=crf, force=force, jobs=jobs or None
    )

    for clip, path, transcoded, error in results:
        if error:
            failed += 1
            if as_json:
                _emit({"original": str(clip), "error": error})
            else:
                fail(f"{clip.name}: {error}")
            continue

        if as_json:
            _emit({
                "path": str(path),
                "original": str(clip),
                "transcoded": transcoded,
            })
        elif transcoded:
            console.print(f"Built proxy: {path}")
        else:
            console.print(f"Already playable: {path}")

    # a partial failure still leaves usable proxies behind, so the exit code
    # reports it without the caller having to treat the run as a total loss
    if failed:
        sys.exit(1)
