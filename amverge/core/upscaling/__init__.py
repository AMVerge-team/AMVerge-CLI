from __future__ import annotations

import importlib
from typing import Any

# Attribute -> (submodule, name in that submodule). Resolved on first access so
# that importing this package does not eagerly pull the ML stack.
_LAZY_ATTRS: dict[str, tuple[str, str]] = {
    # .registry
    'QUALITY_PRESETS': ('.registry', 'QUALITY_PRESETS'),
    'UPSCALE_REGISTRY': ('.registry', 'UPSCALE_REGISTRY'),
    'get_all_model_keys': ('.registry', 'get_all_model_keys'),
    'get_ml_models': ('.registry', 'get_ml_models'),
    'get_model': ('.registry', 'get_model'),
    'get_model_credit': ('.registry', 'get_model_credit'),
    'get_model_scales': ('.registry', 'get_model_scales'),
    'get_models_by_method': ('.registry', 'get_models_by_method'),
    'get_onnx_models': ('.registry', 'get_onnx_models'),
    'get_shader_models': ('.registry', 'get_shader_models'),
    # .weight_loader
    'MODEL_FILES': ('.weight_loader', 'MODEL_FILES'),
    'UPSCALE_MODEL_KEYS': ('.weight_loader', 'UPSCALE_MODEL_KEYS'),
    'download_weights': ('.weight_loader', 'download_weights'),
    'get_weight_path': ('.weight_loader', 'get_weight_path'),
    'is_weight_downloaded': ('.weight_loader', 'is_weight_downloaded'),
    'load_weights_if_available': ('.weight_loader', 'load_weights_if_available'),
    'verify_weight_hash': ('.weight_loader', 'verify_weight_hash'),
    # .monitor
    'SystemMonitor': ('.monitor', 'SystemMonitor'),
    'format_eta': ('.monitor', 'format_eta'),
    'sample_cpu': ('.monitor', 'sample_cpu'),
    'sample_gpu': ('.monitor', 'sample_gpu'),
    # .anime4k
    'ANIME4K_MODE_PRESETS': ('.anime4k', 'ANIME4K_MODE_PRESETS'),
    'download_anime4k_shaders': ('.anime4k', 'download_anime4k_shaders'),
    'get_shader_dir': ('.anime4k', 'get_shader_dir'),
    'is_anime4k_downloaded': ('.anime4k', 'is_anime4k_downloaded'),
    'libplacebo_available': ('.anime4k', 'libplacebo_available'),
    'list_shaders': ('.anime4k', 'list_shaders'),
    # .artcnn
    'download_artcnn': ('.artcnn', 'download_artcnn'),
    'get_artcnn_dir': ('.artcnn', 'get_artcnn_dir'),
    'get_artcnn_path': ('.artcnn', 'get_artcnn_path'),
    'is_artcnn_downloaded': ('.artcnn', 'is_artcnn_downloaded'),
    # .engine
    'UPSCALE_AVAILABLE': ('.engine', 'UPSCALE_AVAILABLE'),
    'upscale_model': ('.engine', 'upscale_model'),
}

# Optional imports: on ImportError these degrade to the values the previous
# try/except blocks used, instead of propagating.
_OPTIONAL_FALLBACKS: dict[str, Any] = {
    'UPSCALE_AVAILABLE': False,
    'upscale_model': None,
}

# Kept reachable as attributes, matching the old eager imports.
_LAZY_SUBMODULES: frozenset[str] = frozenset({
    'anime4k',
    'artcnn',
    'engine',
    'ffmpeg_helpers',
    'monitor',
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
    "UPSCALE_REGISTRY",
    "QUALITY_PRESETS",
    "get_model",
    "get_models_by_method",
    "get_ml_models",
    "get_shader_models",
    "get_onnx_models",
    "get_all_model_keys",
    "get_model_scales",
    "get_model_credit",
    "SystemMonitor",
    "sample_gpu",
    "sample_cpu",
    "format_eta",
    "download_weights",
    "is_weight_downloaded",
    "get_weight_path",
    "verify_weight_hash",
    "load_weights_if_available",
    "MODEL_FILES",
    "UPSCALE_MODEL_KEYS",
    "upscale_model",
    "UPSCALE_AVAILABLE",
    "ANIME4K_MODE_PRESETS",
    "download_anime4k_shaders",
    "is_anime4k_downloaded",
    "libplacebo_available",
    "list_shaders",
    "get_shader_dir",
    "download_artcnn",
    "is_artcnn_downloaded",
    "get_artcnn_path",
    "get_artcnn_dir",
]
