from __future__ import annotations

import importlib
from typing import Any

# Attribute -> (submodule, name in that submodule). Resolved on first access so
# that importing this package does not eagerly pull the ML stack.
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # .depth_map
    'COLMAPS': ('.depth_map', 'COLMAPS'),
    'DEPTH_AVAILABLE': ('.depth_map', 'DEPTH_AVAILABLE'),
    'MODEL_CONFIGS': ('.depth_map', 'MODEL_CONFIGS'),
    'download_model': ('.depth_map', 'download_model'),
    'generate_depth_map': ('.depth_map', 'generate_depth_map'),
    'is_model_downloaded': ('.depth_map', 'is_model_downloaded'),
}

# Optional imports: on ImportError these degrade to the values the previous
# try/except blocks used, instead of propagating.
_OPTIONAL_FALLBACKS: dict[str, Any] = {
    'COLMAPS': {},
    'DEPTH_AVAILABLE': False,
    'MODEL_CONFIGS': {},
    'download_model': None,
    'generate_depth_map': None,
    'is_model_downloaded': None,
}

# Kept reachable as attributes, matching the old eager imports.
_LAZY_SUBMODULES: frozenset[str] = frozenset({
    'depth_map',
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
    try:
        value = getattr(importlib.import_module(module_name, __name__), orig_name)
    except (ImportError, AttributeError):
        if name not in _OPTIONAL_FALLBACKS:
            raise
        value = _OPTIONAL_FALLBACKS[name]
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "generate_depth_map",
    "is_model_downloaded",
    "download_model",
    "DEPTH_AVAILABLE",
    "MODEL_CONFIGS",
    "COLMAPS",
]
