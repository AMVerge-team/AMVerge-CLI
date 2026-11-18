import json
import os
import sys


def _app_data_dirs():
    home = os.path.expanduser("~")
    appdata = os.environ.get("APPDATA", "")
    dirs = []
    if appdata:
        dirs.extend([
            os.path.join(appdata, "app.amverge"),
            os.path.join(appdata, "AMVerge"),
        ])
    if os.name == "nt":
        dirs.append(os.path.join(home, "AppData", "Roaming", "app.amverge"))
    elif sys.platform == "darwin":
        dirs.append(os.path.join(home, "Library", "Application Support", "app.amverge"))
    else:
        dirs.append(os.path.join(home, ".local", "share", "app.amverge"))
    return list(dict.fromkeys(dirs))


def _read_app_paths():
    paths = {}
    for directory in _app_data_dirs():
        paths_file = os.path.join(directory, "paths.json")
        if os.path.isfile(paths_file):
            try:
                with open(paths_file, encoding="utf-8") as handle:
                    payload = json.load(handle)
                if isinstance(payload, dict):
                    for key, value in payload.items():
                        if value is not None and value != "":
                            paths.setdefault(key, value)
            except (OSError, ValueError, TypeError):
                pass
        pointer_file = os.path.join(directory, "ai_storage_path.txt")
        if os.path.isfile(pointer_file):
            try:
                with open(pointer_file, encoding="utf-8") as handle:
                    root = handle.read().strip()
                if root:
                    if not paths.get("ai_config_dir"):
                        paths["ai_config_dir"] = os.path.join(root, "config")
                    if not paths.get("hf_home_dir"):
                        paths["hf_home_dir"] = os.path.join(root, "hf-cache")
            except OSError:
                pass
    return paths


def configure_app_storage() -> None:
    paths = _read_app_paths()
    if not os.environ.get("AMVERGE_CONFIG_DIR", "").strip():
        config = str(paths.get("ai_config_dir") or "").strip()
        if config:
            os.environ["AMVERGE_CONFIG_DIR"] = os.path.abspath(os.path.expanduser(config))
    if (
        not os.environ.get("HF_HOME", "").strip()
        and not os.environ.get("HF_HUB_CACHE", "").strip()
    ):
        hf_home = str(paths.get("hf_home_dir") or "").strip()
        if hf_home:
            os.environ["HF_HOME"] = os.path.abspath(os.path.expanduser(hf_home))


def get_amverge_config_dir():
    configure_app_storage()
    override = os.environ.get("AMVERGE_CONFIG_DIR", "").strip()
    if override:
        return os.path.abspath(os.path.expanduser(override))
    appdata = os.environ.get("APPDATA", "")
    if appdata:
        return os.path.join(appdata, "com.amverge.cli")
    home = os.path.expanduser("~")
    if os.name == "nt":
        return os.path.join(home, "AppData", "Roaming", "com.amverge.cli")
    if sys.platform == "darwin":
        return os.path.join(home, "Library", "Application Support", "com.amverge.cli")
    return os.path.join(home, ".config", "com.amverge.cli")


def get_ffmpeg_dir():
    return os.path.join(get_amverge_config_dir(), "ffmpeg")


def get_models_dir():
    return os.path.join(get_amverge_config_dir(), "models", "upscale")


def get_model_dir(model_key):
    return os.path.join(get_models_dir(), model_key)
