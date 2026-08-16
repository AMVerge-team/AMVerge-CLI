from __future__ import annotations

import importlib
from typing import Any

# Attribute -> (submodule, name in that submodule). Resolved on first access so
# that importing this package does not eagerly pull the ML stack.
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # .edge
    'detect_cuts_by_edge': ('.edge', 'detect_cuts_by_edge'),
    # .keyframe
    'detect_cuts_by_keyframe': ('.keyframe', 'detect_cuts_by_keyframe'),
    # .nelux_runtime
    '_get_nelux_video_reader': ('.nelux_runtime', '_get_nelux_video_reader'),
    # .ai_scene_detection
    'TRANSNET_AVAILABLE': ('.ai_scene_detection', 'TRANSNET_AVAILABLE'),
    'decode_and_detect_scenes': ('.ai_scene_detection', 'decode_and_detect_scenes'),
    'decode_video_frames_nelux': ('.ai_scene_detection', 'decode_video_frames_nelux'),
    'run_model_one_pass': ('.ai_scene_detection', 'run_model_one_pass'),
}

# Kept reachable as attributes, matching the old eager imports.
_LAZY_SUBMODULES: frozenset[str] = frozenset({
    'ai_scene_detection',
    'edge',
    'keyframe',
    'nelux_runtime',
})


def __getattr__(name: str) -> Any:
    if name in _LAZY_SUBMODULES:
        value = importlib.import_module(f'.{name}', __name__)
        globals()[name] = value
        return value
    try:
        module_name, orig_name = _LAZY_ATTRS[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None
    value = getattr(importlib.import_module(module_name, __name__), orig_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


# Mirrors the names the previous eager imports left on this module, so that
# `from ... import *` still yields the same surface.
__all__ = [
    'TRANSNET_AVAILABLE',
    'ai_scene_detection',
    'decode_and_detect_scenes',
    'decode_video_frames_nelux',
    'detect_cuts_by_edge',
    'detect_cuts_by_keyframe',
    'edge',
    'keyframe',
    'nelux_runtime',
    'run_model_one_pass',
]
