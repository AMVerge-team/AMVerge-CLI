"""Fast sampled content fingerprinting for Scene Scout media files.

Uses stdlib ``hashlib.blake2b`` (256-bit digest) over the exact byte length plus up
to three 1 MiB slices (head, midpoint, tail), completing in <5 ms even on multi-GB
video files. Must remain byte-for-byte identical with standalone Scene Scout's
``src/fingerprint.py`` so fingerprints match across both applications.
"""

from __future__ import annotations

import hashlib
import os
import struct
from pathlib import Path

FINGERPRINT_ALGO = "blake2b_sample_v1"
SAMPLE_CHUNK_BYTES = 1_048_576  # 1 MiB


def compute_content_fingerprint(path: str | Path) -> tuple[int, str]:
    """Return ``(file_size, 'blake2b_sample_v1:<64-char-hex>')`` for ``path``."""
    p = Path(path)
    hasher = hashlib.blake2b(digest_size=32)
    with p.open("rb") as f:
        file_size = os.fstat(f.fileno()).st_size
        hasher.update(struct.pack("<Q", file_size))
        if file_size <= 3 * SAMPLE_CHUNK_BYTES:
            while chunk := f.read(SAMPLE_CHUNK_BYTES):
                hasher.update(chunk)
        else:
            hasher.update(f.read(SAMPLE_CHUNK_BYTES))
            mid_offset = (file_size - SAMPLE_CHUNK_BYTES) // 2
            f.seek(mid_offset)
            hasher.update(f.read(SAMPLE_CHUNK_BYTES))
            f.seek(file_size - SAMPLE_CHUNK_BYTES)
            hasher.update(f.read(SAMPLE_CHUNK_BYTES))

    return file_size, f"{FINGERPRINT_ALGO}:{hasher.hexdigest()}"


def verify_content_fingerprint(
    path: str | Path,
    expected_size: int | None,
    expected_fingerprint: str | None,
) -> bool:
    """True when ``path`` exists and matches ``expected_size`` / ``expected_fingerprint``."""
    p = Path(path)
    try:
        actual_size = p.stat().st_size
    except OSError:
        return False

    if expected_size is not None and actual_size != expected_size:
        return False
    if not expected_fingerprint:
        return True

    prefix = f"{FINGERPRINT_ALGO}:"
    if not expected_fingerprint.startswith(prefix):
        return False

    try:
        _, actual_fp = compute_content_fingerprint(p)
    except OSError:
        return False
    return actual_fp == expected_fingerprint
