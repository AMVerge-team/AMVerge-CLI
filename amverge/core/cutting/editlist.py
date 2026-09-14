"""In-place ISOBMFF edit-list patching.

ffmpeg always writes an edit list (``elst``) when it stream-copies into an
mp4/mov container -- even a trivial one covering the whole track -- to
correct for the small decode-ahead every B-frame stream needs (see
``cutting.smart_cut`` module docs). That box already correctly hides
whatever pre-roll a keyframe-backward-snapped ``-ss`` dragged in; what it
does *not* do is shrink to match a ``-t`` duration that overshoots the next
real keyframe. This module rewrites that one field -- the edit list's
declared segment duration -- directly in the file's existing box, in place,
byte-exact, with nothing else touched: no sample is moved, no other offset
in the file shifts, so this is safe to do without re-muxing.

It only ever *rewrites* a field; it never inserts a box. That is a
deliberate scope limit, not an oversight -- inserting a box would grow
``moov`` and require rewriting every chunk offset (``stco``/``co64``) in the
file to compensate, a much bigger and riskier operation. Because ffmpeg
reliably writes an edit list on every stream-copy mp4/mov output (verified
across H.264, HEVC, with and without B-frames), the "no edit list to patch"
case in practice only comes up for inputs this module doesn't recognize;
:func:`patch_trailing_duration` returns False rather than guess, and the
caller falls back to a full re-encode for that one scene.
"""

from __future__ import annotations

import struct
from pathlib import Path

_HEADER = struct.Struct(">I4s")
_LARGESIZE = struct.Struct(">Q")


class _Box:
    __slots__ = ("type", "start", "size", "header_len")

    def __init__(self, box_type: bytes, start: int, size: int, header_len: int) -> None:
        self.type = box_type
        self.start = start
        self.size = size
        self.header_len = header_len

    @property
    def body_start(self) -> int:
        return self.start + self.header_len

    @property
    def body_size(self) -> int:
        return self.size - self.header_len

    @property
    def end(self) -> int:
        return self.start + self.size


def _iter_boxes(f, start: int, end: int):
    """Yield direct-child ``_Box``es within ``[start, end)`` of an open file."""
    pos = start
    while pos + 8 <= end:
        f.seek(pos)
        header = f.read(8)
        if len(header) < 8:
            return
        size, box_type = _HEADER.unpack(header)
        header_len = 8
        if size == 1:
            ext = f.read(8)
            if len(ext) < 8:
                return
            size = _LARGESIZE.unpack(ext)[0]
            header_len = 16
        elif size == 0:
            size = end - pos
        if size < header_len or pos + size > end:
            return
        yield _Box(box_type, pos, size, header_len)
        pos += size


def _find_child(f, box_type: bytes, parent: _Box) -> _Box | None:
    for child in _iter_boxes(f, parent.body_start, parent.end):
        if child.type == box_type:
            return child
    return None


def _find_all_children(f, box_type: bytes, parent: _Box) -> list[_Box]:
    return [b for b in _iter_boxes(f, parent.body_start, parent.end) if b.type == box_type]


def _read_full_box_timescale(f, box: _Box) -> int | None:
    """``timescale`` from ``mvhd`` or ``mdhd`` -- same layout up to that
    field: version(1)+flags(3), then version-dependent creation/modification
    time, then ``timescale`` (always a plain 4-byte field, even in a
    version-1 box)."""
    f.seek(box.body_start)
    version = f.read(1)
    if len(version) != 1:
        return None
    offset = 4 + (16 if version[0] == 1 else 8)  # version/flags + creation/modification time
    f.seek(box.body_start + offset)
    raw = f.read(4)
    if len(raw) != 4:
        return None
    timescale = struct.unpack(">I", raw)[0]
    return timescale or None


def _patch_full_box_duration(f, box: _Box, header_extra: int, new_duration: int) -> bool:
    """Rewrite the ``duration`` field shared by ``mvhd`` and ``tkhd``.

    Both start with version(1)+flags(3), then version-dependent
    creation_time/modification_time (4 or 8 bytes each), then some
    version-independent fixed-size fields before ``duration`` itself (4 or 8
    bytes, matching the creation/modification width): ``mvhd`` has just
    ``timescale`` (always 4 bytes, ``header_extra=4``); ``tkhd`` has
    ``track_ID``+``reserved`` (4+4, ``header_extra=8``). Both express
    ``duration`` in the *movie* timescale, same as an edit list's
    ``segment_duration`` -- one ``new_duration`` value patches all of them.
    """
    f.seek(box.body_start)
    version = f.read(1)
    if len(version) != 1:
        return False
    wide = version[0] == 1
    time_field_size = 8 if wide else 4
    offset = 4 + 2 * time_field_size + header_extra
    if wide:
        packed = struct.pack(">Q", new_duration)
    else:
        if new_duration > 0xFFFFFFFF:
            return False
        packed = struct.pack(">I", new_duration)
    f.seek(box.body_start + offset)
    f.write(packed)
    return True


def _patch_elst_duration(f, elst: _Box, new_duration: int) -> bool:
    """Rewrite the one content-referencing entry's duration in an edit list.

    ffmpeg has produced two shapes so far: a single entry (plain ``-ss``/``-c
    copy``), and two entries from the ``segment`` muxer, where the first is a
    fixed leading "empty edit" (``media_time == -1``, a sync pad, not
    something this trim should touch) and the second references real media.
    Either way there is exactly one entry whose ``media_time != -1``; that is
    the one this rewrites. More than one (or none) is an edit-list shape this
    module doesn't recognize, and it declines rather than guess which entry
    -- or entries -- actually govern the displayed duration.
    """
    f.seek(elst.body_start)
    header = f.read(8)  # version/flags (4) + entry_count (4)
    if len(header) != 8:
        return False
    version = header[0]
    entry_count = struct.unpack(">I", header[4:8])[0]
    if entry_count < 1:
        return False
    dur_size = 8 if version == 1 else 4
    time_fmt = ">q" if version == 1 else ">i"
    entry_size = dur_size * 2 + 4  # duration + media_time + 2x16.16 media_rate
    entries_start = elst.body_start + 8

    content_entry_offset = None
    for i in range(entry_count):
        entry_off = entries_start + i * entry_size
        f.seek(entry_off + dur_size)  # media_time immediately follows duration
        raw_media_time = f.read(dur_size)
        if len(raw_media_time) != dur_size:
            return False
        media_time = struct.unpack(time_fmt, raw_media_time)[0]
        if media_time != -1:
            if content_entry_offset is not None:
                return False
            content_entry_offset = entry_off

    if content_entry_offset is None:
        return False
    if dur_size == 8:
        packed = struct.pack(">Q", new_duration)
    else:
        if new_duration > 0xFFFFFFFF:
            return False
        packed = struct.pack(">I", new_duration)
    f.seek(content_entry_offset)
    f.write(packed)
    return True


def patch_trailing_duration(path: Path, duration_sec: float) -> bool:
    """Rewrite every duration field in ``path`` that governs displayed length.

    That's ``mvhd.duration`` (the file's own overall declared duration),
    each track's ``tkhd.duration`` (same movie timescale, same reason), each
    track's edit-list duration (what a player actually stops display at),
    and -- despite describing the track's own *media* rather than the
    presentation -- each track's ``mdhd.duration`` too: ffprobe's own
    ``stream=duration``/``format=duration`` were observed reading directly
    from it rather than the edit list for one of the two box shapes ffmpeg
    produces (the ``segment`` muxer's, which adds a leading empty-edit
    entry -- see ``_patch_elst_duration``), so leaving it stale reproduces
    exactly the kind of tool-visible duration mismatch this exists to close.
    All four express the same real, unchanged content; only their metadata
    is rewritten, each in its own box's own timescale.

    Leaves every other byte in the file untouched -- no sample moves, no
    other box resizes. Returns True if every field above was patched; False
    if the file's structure doesn't match what this module knows how to
    patch (missing/unexpected boxes, an edit list with more than one
    content-referencing entry, etc.), in which case the caller should not
    trust the file's declared duration and should fall back to a re-encode
    instead.
    """
    try:
        with open(path, "r+b") as f:
            size = f.seek(0, 2)
            f.seek(0)
            ftyp = None
            moov = None
            for top in _iter_boxes(f, 0, size):
                if top.type == b"ftyp":
                    ftyp = top
                elif top.type == b"moov":
                    moov = top
            if moov is None or ftyp is None:
                return False

            mvhd = _find_child(f, b"mvhd", moov)
            if mvhd is None:
                return False
            movie_timescale = _read_full_box_timescale(f, mvhd)
            if not movie_timescale:
                return False
            new_duration = round(duration_sec * movie_timescale)

            traks = _find_all_children(f, b"trak", moov)
            if not traks:
                return False

            if not _patch_full_box_duration(f, mvhd, header_extra=4, new_duration=new_duration):
                return False

            for trak in traks:
                tkhd = _find_child(f, b"tkhd", trak)
                if tkhd is None:
                    return False
                if not _patch_full_box_duration(f, tkhd, header_extra=8, new_duration=new_duration):
                    return False

                mdia = _find_child(f, b"mdia", trak)
                if mdia is None:
                    return False
                mdhd = _find_child(f, b"mdhd", mdia)
                if mdhd is None:
                    return False
                media_timescale = _read_full_box_timescale(f, mdhd)
                if not media_timescale:
                    return False
                media_new_duration = round(duration_sec * media_timescale)
                if not _patch_full_box_duration(f, mdhd, header_extra=4, new_duration=media_new_duration):
                    return False

                edts = _find_child(f, b"edts", trak)
                if edts is None:
                    return False
                elst = _find_child(f, b"elst", edts)
                if elst is None:
                    return False
                if not _patch_elst_duration(f, elst, new_duration):
                    return False
            return True
    except OSError:
        return False
