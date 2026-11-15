from ..infra.model_store import load_registry_json

_registry_data = load_registry_json(__file__)

_sources = _registry_data["_source"]


def _build_model_entry(key, raw):
    entry = dict(raw)
    entry["key"] = key
    if "url" not in entry and "file" in entry:
        base = _sources.get("deadframes", "")
        if base:
            entry["url"] = base + "/" + entry["file"]
    return entry


DEADFRAMES_REGISTRY = {
    key: _build_model_entry(key, raw)
    for key, raw in _registry_data.items()
    if not key.startswith("_")
}


def get_model(key):
    return DEADFRAMES_REGISTRY.get(key)


def get_models_by_method(method=None):
    if method:
        return {k: v for k, v in DEADFRAMES_REGISTRY.items() if v["method"] == method}
    return dict(DEADFRAMES_REGISTRY)


def get_heuristic_models():
    return get_models_by_method("heuristic")


def get_all_model_keys():
    return list(DEADFRAMES_REGISTRY.keys())


def get_model_credit(key):
    entry = DEADFRAMES_REGISTRY.get(key)
    return entry.get("credit", "") if entry else ""
