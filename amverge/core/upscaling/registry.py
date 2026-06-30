MODELS_URL = "https://github.com/moongetsu/AniSmooth-Models/releases/download/"

UPSCALE_REGISTRY = {
    "adore": {
        "key": "adore",
        "method": "ml",
        "name": "Adore",
        "scales": [2, 4],
        "credit": "based on AniSmooth by moongetsu",
        "description": "High quality anime upscaler",
        "category": "upscale",
        "file": "adore.pth",
        "url": MODELS_URL + "upscale/adore.pth",
        "hash": "443378bdc6db6cf4a75eea61ee7afc78b2c4b6a4d3b3981a40ff61f38bbc8f1a",
    },
    "shufflecugan": {
        "key": "shufflecugan",
        "method": "ml",
        "name": "ShuffleCUGAN",
        "scales": [2, 4],
        "credit": "based on AniSmooth by moongetsu",
        "description": "General purpose anime upscaler",
        "category": "upscale",
        "file": "sudo_shuffle_cugan_9.584.969.pth",
        "url": MODELS_URL + "upscale/sudo_shuffle_cugan_9.584.969.pth",
        "hash": "88a6d89f04eaf27a9f7b60937857768a6bc04fb360670bd9951ef533acab0616",
    },
    "fallin_soft": {
        "key": "fallin_soft",
        "method": "ml",
        "name": "Fallin Soft",
        "scales": [2, 4],
        "credit": "based on AniSmooth by moongetsu",
        "description": "Soft anime upscaler, preserves fine details",
        "category": "upscale",
        "file": "Fallin_soft.pth",
        "url": MODELS_URL + "upscale/Fallin_soft.pth",
        "hash": "910aa56a9a1187df97c3284177da1bc66836679350b2613191340734937e9960",
    },
    "fallin_strong": {
        "key": "fallin_strong",
        "method": "ml",
        "name": "Fallin Strong",
        "scales": [2, 4],
        "credit": "based on AniSmooth by moongetsu",
        "description": "Strong anime upscaler with more sharpening",
        "category": "upscale",
        "file": "Fallin_strong.pth",
        "url": MODELS_URL + "upscale/Fallin_strong.pth",
        "hash": "14b8415199aa66a6507725408a66758ba2bff9286736f19f7f07524efd821a56",
    },
    "anime4k": {
        "key": "anime4k",
        "method": "shader",
        "name": "Anime4K",
        "scales": [2, 4],
        "credit": "by bloc97 (MIT License)",
        "description": "GPU shader-based upscaler, no ML deps",
        "modes": ["light", "medium", "strong"],
        "default_mode": "medium",
        "download_url": "https://github.com/bloc97/Anime4K/releases/download/v4.0.1/Anime4K_v4.0.zip",
    },
    "C4F16": {
        "key": "C4F16",
        "method": "onnx",
        "name": "ArtCNN C4F16",
        "scales": [2],
        "credit": "by Artoriuz",
        "description": "Lightweight real-time anime upscaler (12K params)",
        "file": "ArtCNN_C4F16.onnx",
        "url": "https://github.com/Artoriuz/ArtCNN/releases/download/v1.6.2/ArtCNN_C4F16.onnx",
        "input_channels": 1,
    },
    "C4F32": {
        "key": "C4F32",
        "method": "onnx",
        "name": "ArtCNN C4F32",
        "scales": [2],
        "credit": "by Artoriuz",
        "description": "Balanced real-time anime upscaler (48K params)",
        "file": "ArtCNN_C4F32.onnx",
        "url": "https://github.com/Artoriuz/ArtCNN/releases/download/v1.6.2/ArtCNN_C4F32.onnx",
        "input_channels": 1,
    },
    "R8F64": {
        "key": "R8F64",
        "method": "onnx",
        "name": "ArtCNN R8F64",
        "scales": [2],
        "credit": "by Artoriuz",
        "description": "High quality non-real-time anime upscaler (926K params)",
        "file": "ArtCNN_R8F64.onnx",
        "url": "https://github.com/Artoriuz/ArtCNN/releases/download/v1.6.2/ArtCNN_R8F64.onnx",
        "input_channels": 1,
    },
}

QUALITY_PRESETS = {
    "archival": {"crf": 14, "x264": "slow",      "tune": "animation"},
    "high":     {"crf": 17, "x264": "slow",      "tune": "animation"},
    "balanced": {"crf": 20, "x264": "medium",    "tune": "animation"},
    "fast":     {"crf": 22, "x264": "veryfast",  "tune": "animation"},
    "draft":    {"crf": 26, "x264": "ultrafast", "tune": "animation"},
}


def get_model(key):
    return UPSCALE_REGISTRY.get(key)


def get_models_by_method(method=None):
    if method:
        return {k: v for k, v in UPSCALE_REGISTRY.items() if v["method"] == method}
    return dict(UPSCALE_REGISTRY)


def get_ml_models():
    return get_models_by_method("ml")


def get_shader_models():
    return get_models_by_method("shader")


def get_onnx_models():
    return get_models_by_method("onnx")


def get_all_model_keys():
    return list(UPSCALE_REGISTRY.keys())


def get_model_scales(key):
    entry = UPSCALE_REGISTRY.get(key)
    return entry["scales"] if entry else [2, 4]


def get_model_credit(key):
    entry = UPSCALE_REGISTRY.get(key)
    return entry.get("credit", "") if entry else ""
