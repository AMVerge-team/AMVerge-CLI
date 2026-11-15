from ..infra.model_store import QUALITY_PRESETS as _BASE_PRESETS, load_registry_json

_registry_data = load_registry_json(__file__)

QUALITY_PRESETS = {
    name: {**preset, "tune": "animation"}
    for name, preset in _BASE_PRESETS.items()
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


def get_pervfi_models():
    return {k: v for k, v in INTERPOLATION_REGISTRY.items() if v["method"] == "pervfi"}


def get_all_model_keys():
    return list(INTERPOLATION_REGISTRY.keys())


def get_model_credit(key):
    entry = INTERPOLATION_REGISTRY.get(key)
    return entry.get("credit", "") if entry else ""
