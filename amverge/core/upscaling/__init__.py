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

from .monitor import SystemMonitor, sample_gpu, sample_cpu, format_eta

from .anime4k import (
    ANIME4K_MODE_PRESETS,
    download_anime4k_shaders,
    is_anime4k_downloaded,
    libplacebo_available,
    list_shaders,
    get_shader_dir,
)
from .artcnn import (
    download_artcnn,
    is_artcnn_downloaded,
    get_artcnn_path,
    get_artcnn_dir,
)

try:
    from .engine import (
        upscale_model,
        UPSCALE_AVAILABLE,
    )
except ImportError:
    upscale_model = None
    UPSCALE_AVAILABLE = False

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
