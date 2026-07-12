try:
    from .depth_map import (
        generate_depth_map,
        is_model_downloaded,
        download_model,
        DEPTH_AVAILABLE,
        MODEL_CONFIGS,
        COLMAPS,
    )
except ImportError:
    generate_depth_map = None
    is_model_downloaded = None
    download_model = None
    DEPTH_AVAILABLE = False
    MODEL_CONFIGS = {}
    COLMAPS = {}

__all__ = [
    "generate_depth_map",
    "is_model_downloaded",
    "download_model",
    "DEPTH_AVAILABLE",
    "MODEL_CONFIGS",
    "COLMAPS",
]
