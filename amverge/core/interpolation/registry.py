import json
from pathlib import Path


def _load_registry():
    json_path = Path(__file__).parent / "registry.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


_registry_data = _load_registry()

QUALITY_PRESETS = {
    "archival": {"crf": 14, "x264": "slow", "tune": "animation"},
    "high":     {"crf": 17, "x264": "slow", "tune": "animation"},
    "balanced": {"crf": 20, "x264": "medium", "tune": "animation"},
    "fast":     {"crf": 22, "x264": "veryfast", "tune": "animation"},
    "draft":    {"crf": 26, "x264": "ultrafast", "tune": "animation"},
}

_sources = _registry_data["_source"]


def _build_model_entry(key, raw):
    entry = dict(raw)
    entry["key"] = key
    if "url" not in entry:
        base = _sources.get("interpolation", "")
        entry["url"] = base + "/" + entry["file"]
    return entry


INTERPOLATION_REGISTRY = {
    key: _build_model_entry(key, raw)
    for key, raw in _registry_data.items()
    if not key.startswith("_")
}


def get_model(key):
    return INTERPOLATION_REGISTRY.get(key)


def get_rife_models():
    return {k: v for k, v in INTERPOLATION_REGISTRY.items() if v["method"] == "rife"}


def get_all_model_keys():
    return list(INTERPOLATION_REGISTRY.keys())


def get_model_credit(key):
    entry = INTERPOLATION_REGISTRY.get(key)
    return entry.get("credit", "") if entry else ""
