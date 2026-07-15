from .registry import (
    DEADFRAMES_REGISTRY,
    get_model,
    get_models_by_method,
    get_heuristic_models,
    get_all_model_keys,
    get_model_credit,
)

from .weight_loader import (
    download_weights,
    is_weight_downloaded,
    get_weight_path,
    verify_weight_hash,
)

try:
    from .engine import (
        run_deadframes,
        DEADFRAMES_AVAILABLE,
        DeadFrameDetector,
    )
except ImportError:
    run_deadframes = None
    DEADFRAMES_AVAILABLE = False
    DeadFrameDetector = None

__all__ = [
    "DEADFRAMES_REGISTRY",
    "get_model",
    "get_models_by_method",
    "get_heuristic_models",
    "get_all_model_keys",
    "get_model_credit",
    "download_weights",
    "is_weight_downloaded",
    "get_weight_path",
    "verify_weight_hash",
    "run_deadframes",
    "DEADFRAMES_AVAILABLE",
    "DeadFrameDetector",
]
