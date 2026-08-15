import json
import os
from typing import Optional

from ..infra.config import get_amverge_config_dir


def _get_pipelines_dir():
    d = os.path.join(get_amverge_config_dir(), "pipelines")
    os.makedirs(d, exist_ok=True)
    return d


def list_presets():
    d = _get_pipelines_dir()
    presets = []
    for f in sorted(os.listdir(d)):
        if f.endswith(".json"):
            presets.append(f[:-5])
    return presets


def load_preset(name: str) -> Optional[dict]:
    path = os.path.join(_get_pipelines_dir(), f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_preset(name: str, config: dict):
    path = os.path.join(_get_pipelines_dir(), f"{name}.json")
    config["name"] = name
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)


def delete_preset(name: str) -> bool:
    path = os.path.join(_get_pipelines_dir(), f"{name}.json")
    if os.path.exists(path):
        os.unlink(path)
        return True
    return False


def _resolve_top_level_config(config: dict) -> dict:
    if "deadframes" not in config:
        config["deadframes"] = {}
    if "upscale" not in config:
        config["upscale"] = {}
    if "interpolate" not in config:
        config["interpolate"] = {}
    return config
