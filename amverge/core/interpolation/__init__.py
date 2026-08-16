from __future__ import annotations

import importlib
from typing import Any

# Attribute -> (submodule, name in that submodule). Resolved on first access so
# that importing this package does not eagerly pull the ML stack.
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # .flowframes
    'FLOWFRAMES_MODELS': ('.flowframes', 'FLOWFRAMES_MODELS'),
    'FLOWFRAMES_VERSION': ('.flowframes', 'FLOWFRAMES_VERSION'),
    'cancel_flowframes': ('.flowframes', 'cancel_flowframes'),
    'flowframes_available': ('.flowframes', 'flowframes_available'),
    'get_flowframes_path': ('.flowframes', 'get_flowframes_path'),
    'is_flowframes_model_installed': ('.flowframes', 'is_flowframes_model_installed'),
    'run_flowframes': ('.flowframes', 'run_flowframes'),
    'set_flowframes_path': ('.flowframes', 'set_flowframes_path'),
    # .registry
    'INTERPOLATION_REGISTRY': ('.registry', 'INTERPOLATION_REGISTRY'),
    'QUALITY_PRESETS': ('.registry', 'QUALITY_PRESETS'),
    'get_all_model_keys': ('.registry', 'get_all_model_keys'),
    'get_model': ('.registry', 'get_model'),
    'get_model_credit': ('.registry', 'get_model_credit'),
    'get_pervfi_models': ('.registry', 'get_pervfi_models'),
    'get_rife_models': ('.registry', 'get_rife_models'),
    # .weight_loader
    'download_weights': ('.weight_loader', 'download_weights'),
    'get_weight_path': ('.weight_loader', 'get_weight_path'),
    'is_weight_downloaded': ('.weight_loader', 'is_weight_downloaded'),
    'load_weights_if_available': ('.weight_loader', 'load_weights_if_available'),
    'verify_weight_hash': ('.weight_loader', 'verify_weight_hash'),
    # .engine
    'INTERPOLATION_AVAILABLE': ('.engine', 'INTERPOLATION_AVAILABLE'),
    'interpolate_video': ('.engine', 'interpolate_video'),
}

# Optional imports: on ImportError these degrade to the values the previous
# try/except blocks used, instead of propagating.
_OPTIONAL_FALLBACKS: dict[str, Any] = {
    'INTERPOLATION_AVAILABLE': False,
    'interpolate_video': None,
}

# Kept reachable as attributes, matching the old eager imports.
_LAZY_SUBMODULES: frozenset[str] = frozenset({
    'engine',
    'flowframes',
    'registry',
    'weight_loader',
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
    "flowframes_available",
    "run_flowframes",
    "cancel_flowframes",
    "set_flowframes_path",
    "get_flowframes_path",
    "FLOWFRAMES_VERSION",
    "FLOWFRAMES_MODELS",
    "is_flowframes_model_installed",
    "INTERPOLATION_REGISTRY",
    "QUALITY_PRESETS",
    "get_model",
    "get_rife_models",
    "get_pervfi_models",
    "get_all_model_keys",
    "get_model_credit",
    "download_weights",
    "is_weight_downloaded",
    "get_weight_path",
    "verify_weight_hash",
    "load_weights_if_available",
    "interpolate_video",
    "INTERPOLATION_AVAILABLE",
]
