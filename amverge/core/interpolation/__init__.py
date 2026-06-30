from .flowframes import (
    flowframes_available,
    run_flowframes,
    cancel_flowframes,
    set_flowframes_path,
    get_flowframes_path,
    FLOWFRAMES_VERSION,
)

from .registry import (
    INTERPOLATION_REGISTRY,
    QUALITY_PRESETS,
    get_model,
    get_rife_models,
    get_all_model_keys,
    get_model_credit,
)

from .weight_loader import (
    download_weights,
    is_weight_downloaded,
    get_weight_path,
    verify_weight_hash,
    load_weights_if_available,
)

try:
    from .engine import (
        interpolate_video,
        INTERPOLATION_AVAILABLE,
    )
except ImportError:
    interpolate_video = None
    INTERPOLATION_AVAILABLE = False

__all__ = [
    "flowframes_available",
    "run_flowframes",
    "cancel_flowframes",
    "set_flowframes_path",
    "get_flowframes_path",
    "FLOWFRAMES_VERSION",
    "INTERPOLATION_REGISTRY",
    "QUALITY_PRESETS",
    "get_model",
    "get_rife_models",
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
