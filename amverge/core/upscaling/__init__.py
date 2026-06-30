from .registry import (
    UPSCALE_REGISTRY,
    QUALITY_PRESETS,
    get_model,
    get_models_by_method,
    get_ml_models,
    get_shader_models,
    get_onnx_models,
    get_all_model_keys,
    get_model_scales,
    get_model_credit,
)

from .weight_loader import (
    download_weights,
    is_weight_downloaded,
    get_weight_path,
    verify_weight_hash,
    load_weights_if_available,
    MODEL_FILES,
    UPSCALE_MODEL_KEYS,
)

try:
    from .upscale import (
        upscale_video,
        UPSCALE_AVAILABLE,
    )
except ImportError:
    upscale_video = None
    UPSCALE_AVAILABLE = False

from .anime4k import (
    upscale_video_anime4k,
    ANIME4K_MODE_PRESETS,
    ANIME4K_SHADER_FILES,
)

from .artcnn import (
    upscale_video_artcnn,
    ARTCNN_MODELS,
)

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
    "download_weights",
    "is_weight_downloaded",
    "get_weight_path",
    "verify_weight_hash",
    "load_weights_if_available",
    "MODEL_FILES",
    "UPSCALE_MODEL_KEYS",
    "upscale_video",
    "UPSCALE_AVAILABLE",
    "upscale_video_anime4k",
    "ANIME4K_MODE_PRESETS",
    "ANIME4K_SHADER_FILES",
    "upscale_video_artcnn",
    "ARTCNN_MODELS",
]
