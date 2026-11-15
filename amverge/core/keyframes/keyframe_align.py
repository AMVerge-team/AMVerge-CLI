from __future__ import annotations

"""Keyframe timestamp extraction.

Extracts keyframe timestamps via PyAV packet demux (no frame decode).
``cutting.smart_cut.snap_range_to_keyframes`` uses this list directly to
widen a scene's cut outward to its enclosing keyframes.

.. warning::
    Do not set ``stream.discard`` on the demuxed stream, however tempting the
    "skip non-key packets" framing looks. On B-frame streams it corrupts the
    ``pts`` PyAV reports for the packets it does keep -- confirmed against
    ffprobe ground truth off by whole frames in either direction (a keyframe
    at the true 2.100s came back as 2.07, one at 1.500s as 1.57). Downstream,
    ``smart_cut`` trusts this list to name the exact keyframe a stream-copy
    should snap to; an undercut value sends ffmpeg's backward-snapping
    ``-ss`` to the *previous* real keyframe instead, silently dragging the
    prior scene's frames into the cut. See the ``02_forward_match_bleed_BUG``
    / ``06_head_forward_match_bleed_BUG`` fixtures in
    ``examples/cutting/synthetic/fixtures``.

Usage:
    >>> from amverge.core.keyframes.keyframe_align import get_keyframe_timestamps_pyav
    >>> keyframes = get_keyframe_timestamps_pyav("episode.mp4")
"""

import av


def get_keyframe_timestamps_pyav(video_path: str) -> list[float]:
    """Extract keyframe timestamps using PyAV packet demux.

    Demuxes every packet (no frame decode occurs either way -- decode only
    happens on ``.decode()``, never on plain iteration) and keeps the ones
    flagged as keyframes. Returns deduplicated, sorted timestamps in seconds
    kept at their packet-level precision.

    Deliberately does NOT set ``stream.discard`` to skip non-key packets:
    that corrupts the ``pts`` PyAV reports for the packets it keeps on
    B-frame streams (see the module warning above). Demuxing everything is
    still cheap -- it is header/index parsing, not decoding -- and it is the
    only way to get timestamps that agree with ffprobe.

    Args:
        video_path: Path to the source video file.

    Returns:
        Sorted list of unique keyframe timestamps in seconds.

    Example:
        >>> kf = get_keyframe_timestamps_pyav("episode.mp4")
        >>> print(f"{len(kf)} keyframes, first at {kf[0]}s")
    """
    keyframe_times: list[float] = []
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        for packet in container.demux(stream):
            if not packet.is_keyframe:
                continue
            ts = packet.pts if packet.pts is not None else packet.dts
            if ts is None:
                continue
            keyframe_times.append(float(ts * packet.time_base))
    return sorted(set(keyframe_times))


def get_open_gop_keyframes(video_path: str) -> list[float]:
    """Timestamps of the keyframes that have leading pictures: pictures
    stored after the keyframe that display before it. In HEVC those are the
    RASL pictures of an open-GOP CRA, which also reference the GOP before
    it, so a decoder that starts at the CRA discards them. A stream copy
    split at such a keyframe loses those frames from both clips.

    x265 (open GOP by default) gives them only to keyframes it places by
    interval in the middle of a long shot; one at a scene change ends the
    previous mini-GOP on a P-frame and has none. On a real 24-minute x265
    episode that was 13 of 321 keyframes. Keyframe detection skips these
    (see ``generate_keyframes(skip_open_gop=True)``).

    Also matches RADL pictures, which are decodable from the keyframe alone;
    treating those as unsafe only costs a cut point, never a wrong clip.
    """
    open_gop: list[float] = []
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        last_key = None
        for packet in container.demux(stream):
            if packet.pts is None:
                continue
            if packet.is_keyframe:
                last_key = packet.pts
                continue
            if last_key is not None and packet.pts < last_key:
                open_gop.append(float(last_key * stream.time_base))
                last_key = None
    return open_gop


def scan_keyframes(video_path: str) -> tuple[list[float], list[float]]:
    """``(get_keyframe_timestamps_pyav, get_open_gop_keyframes)`` from one
    demux pass instead of two: both read every packet of the file, which is
    most of the cost on a full episode."""
    keyframe_times: list[float] = []
    open_gop: list[float] = []
    with av.open(video_path) as container:
        stream = container.streams.video[0]
        last_key = None
        for packet in container.demux(stream):
            if packet.is_keyframe:
                ts = packet.pts if packet.pts is not None else packet.dts
                if ts is not None:
                    keyframe_times.append(float(ts * packet.time_base))
                if packet.pts is not None:
                    last_key = packet.pts
                continue
            if packet.pts is None:
                continue
            if last_key is not None and packet.pts < last_key:
                open_gop.append(float(last_key * stream.time_base))
                last_key = None
    return sorted(set(keyframe_times)), open_gop


def cached_scan_keyframes(video_path: str, cache_dir: str | None) -> tuple[list[float], list[float]]:
    """:func:`scan_keyframes` through an on-disk cache in ``cache_dir`` (none:
    no cache). Entries are keyed by path, size and mtime, so an edited or
    replaced file is rescanned. The caller owns the folder's lifetime (the
    app clears it on close and launch)."""
    import hashlib
    import json
    import os
    import threading
    from pathlib import Path

    if not cache_dir:
        return scan_keyframes(video_path)
    try:
        st = os.stat(video_path)
    except OSError:
        return scan_keyframes(video_path)
    key = f"{os.path.abspath(video_path)}|{st.st_size}|{st.st_mtime_ns}"
    entry = Path(cache_dir) / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()[:32]}.json"
    try:
        data = json.loads(entry.read_text(encoding="utf-8"))
        if data.get("key") == key:
            return list(data["keyframes"]), list(data["open_gop"])
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass
    keyframes, open_gop = scan_keyframes(video_path)
    try:
        entry.parent.mkdir(parents=True, exist_ok=True)
        tmp = entry.with_name(f"{entry.stem}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps({"key": key, "keyframes": keyframes, "open_gop": open_gop}), encoding="utf-8")
        os.replace(tmp, entry)
    except OSError:
        pass
    return keyframes, open_gop
