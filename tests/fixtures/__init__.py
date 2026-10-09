from __future__ import annotations

import pytest

from .known_bugs import known_bug
from .specs import BY_NAME, FIXTURES, MediaSpec


def media_params(*names: str, xfail: dict[str, str] | None = None) -> list:
    specs = [BY_NAME[n] for n in names] if names else list(FIXTURES)
    xfail = xfail or {}
    return [
        pytest.param(spec, id=spec.name, marks=[known_bug(xfail[spec.name])] if spec.name in xfail else [])
        for spec in specs
    ]


__all__ = ["BY_NAME", "FIXTURES", "MediaSpec", "known_bug", "media_params"]
