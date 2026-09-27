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
