import os

from ..infra.config import get_models_dir
from ..infra.model_store import compute_sha256, download_with_resume
from .registry import get_ml_models, get_model

WEIGHTS_DIR = get_models_dir()

MODEL_FILES = {
    key: (entry.get("category", ""), entry["file"])
    for key, entry in get_ml_models().items()
}

UPSCALE_MODEL_KEYS = list(MODEL_FILES.keys())

MODEL_HASHES = {
    key: entry["hash"]
    for key, entry in get_ml_models().items()
    if "hash" in entry
}


def _model_filename(model_key):
    entry = get_model(model_key)
    if entry is None or "file" not in entry:
        raise ValueError("Unknown model key: " + model_key)
    return entry["file"]


def _model_url(model_key):
    entry = get_model(model_key)
    if entry is None:
        raise ValueError("Unknown model key: " + model_key)
    return entry["url"]


def ensure_weights_dir():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    return WEIGHTS_DIR


def get_weight_path(model_key):
    subdir = model_key
    filename = _model_filename(model_key)
    return os.path.join(WEIGHTS_DIR, subdir, filename)


def is_weight_downloaded(model_key):
    try:
        return os.path.exists(get_weight_path(model_key))
    except ValueError:
        return False


def verify_weight_hash(model_key, weight_path):
    entry = get_model(model_key)
    expected_hash = entry.get("hash") if entry else None
    if expected_hash is None:
        raise ValueError(f"No verified hash registered for {model_key}")
    calculated_hash = compute_sha256(weight_path)
    if calculated_hash != expected_hash:
        raise RuntimeError(
            f"Weight hash mismatch for {model_key}. "
            f"Expected: {expected_hash[:12]}..., Got: {calculated_hash[:12]}..."
        )
    return True


def download_weights(model_key, force=False, retries=3, progress_cb=None):
    entry = get_model(model_key)
    if entry is None:
        raise ValueError("Unknown model key: " + model_key)

    filename = entry["file"]
    folder_path = os.path.join(WEIGHTS_DIR, model_key)
    dest = os.path.join(folder_path, filename)

    if os.path.exists(dest) and not force:
        return True

    temp_folder = os.path.join(folder_path, "TEMP")
    temp_path = os.path.join(temp_folder, filename)

    try:
        return download_with_resume(
            entry["url"],
            dest,
            temp_path=temp_path,
            expected_hash=entry.get("hash"),
            retries=retries,
            progress_cb=progress_cb,
            progress_formatter=lambda pct, _dl, _tot: f"Downloading {filename} {pct}%",
            done_message=f"Downloading {filename} 100%",
            cleanup_temp_dir=temp_folder,
            remove_dest_on_hash_mismatch=True,
        )
    except Exception:
        return False


def _remap_state_dict_keys(state_dict, model):
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state_dict.keys())

    if ckpt_keys & model_keys:
        overlap = len(ckpt_keys & model_keys)
        if overlap == len(model_keys) or overlap == len(ckpt_keys):
            return state_dict

    prefixes_to_strip = ["module.flownet.", "module.", "flownet."]
    prefixes_to_add = ["flownet.", "module.", "module.flownet."]

    for prefix in prefixes_to_strip:
        remapped = {}
        for k, v in state_dict.items():
            new_key = k[len(prefix):] if k.startswith(prefix) else k
            remapped[new_key] = v
        overlap = len(set(remapped.keys()) & model_keys)
        if overlap > len(ckpt_keys & model_keys):
            return remapped

    for prefix in prefixes_to_add:
        remapped = {}
        for k, v in state_dict.items():
            new_key = prefix + k
            remapped[new_key] = v
        overlap = len(set(remapped.keys()) & model_keys)
        if overlap > len(ckpt_keys & model_keys):
            return remapped

    return state_dict


def load_weights_if_available(model, model_key, device=None):
    import torch

    try:
        weight_path = get_weight_path(model_key)
    except ValueError:
        return False

    if not os.path.exists(weight_path):
        return False

    if not verify_weight_hash(model_key, weight_path):
        raise RuntimeError(
            f"Model integrity verification failed for {model_key}. "
            "File is corrupted or untrusted."
        )

    try:
        state_dict = torch.load(weight_path, map_location=device or "cpu",
                                weights_only=True)

        state_dict = _remap_state_dict_keys(state_dict, model)

        model_param_count = len(model.state_dict())
        try:
            model.load_state_dict(state_dict, strict=True)
            return True
        except RuntimeError:
            pass

        result = model.load_state_dict(state_dict, strict=False)
        loaded_count = model_param_count - len(result.missing_keys)

        threshold = int(model_param_count * 0.5)
        if loaded_count < threshold:
            raise RuntimeError(
                f"Only {loaded_count}/{model_param_count} weights loaded "
                f"for {model_key} (threshold: {threshold}). "
                "Model will produce garbage output."
            )

        return True
    except Exception as e:
        raise RuntimeError(f"Failed to load weights for {model_key}: {e}")
