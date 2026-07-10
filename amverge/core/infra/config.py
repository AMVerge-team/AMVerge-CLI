import os


def get_amverge_config_dir():
    appdata = os.environ.get("APPDATA", "")
    if appdata:
        return os.path.join(appdata, "com.amverge.cli")
    home = os.path.expanduser("~")
    if os.name == "nt":
        return os.path.join(home, "AppData", "Roaming", "com.amverge.cli")
    return os.path.join(home, ".config", "com.amverge.cli")


def get_ffmpeg_dir():
    return os.path.join(get_amverge_config_dir(), "ffmpeg")


def get_models_dir():
    return os.path.join(get_amverge_config_dir(), "models", "upscale")


def get_model_dir(model_key):
    return os.path.join(get_models_dir(), model_key)
