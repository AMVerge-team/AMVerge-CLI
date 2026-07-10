import json
from pathlib import Path


def _load_registry():
    json_path = Path(__file__).parent / "registry.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


_registry_data = _load_registry()

QUALITY_PRESETS = _registry_data["presets"]
_sources = _registry_data["_source"]


def _build_model_entry(key, raw):
    entry = dict(raw)
    entry["key"] = key

    if "url" not in entry:
        if entry["method"] == "ml":
            category = entry.get("category", "upscale")
            base = _sources.get("ml", "")
            entry["url"] = base + category + "/" + entry["file"]
        elif entry["method"] == "onnx":
            base = _sources.get("artcnn", "")
            entry["url"] = base + entry["file"]
        elif entry["method"] == "shader":
            entry["download_url"] = _sources.get("anime4k", "")

    return entry


UPSCALE_REGISTRY = {
    key: _build_model_entry(key, raw)
    for key, raw in _registry_data["models"].items()
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
