import json
import os
from amverge.core.infra.config import (
    configure_app_storage,
    get_amverge_config_dir,
    get_ffmpeg_dir,
    get_models_dir,
)
from amverge.core.depth.registry import _get_depth_models_dir


def test_config_dir_env_override(monkeypatch, tmp_path):
    custom = tmp_path / "amverge-ai" / "config"
    monkeypatch.setenv("AMVERGE_CONFIG_DIR", str(custom))

    assert get_amverge_config_dir() == str(custom)
    assert get_models_dir() == os.path.join(str(custom), "models", "upscale")
    assert get_ffmpeg_dir() == os.path.join(str(custom), "ffmpeg")
    assert _get_depth_models_dir() == os.path.join(str(custom), "models", "depth")


def test_config_dir_falls_back_when_unset(monkeypatch, tmp_path):
    monkeypatch.delenv("AMVERGE_CONFIG_DIR", raising=False)
    monkeypatch.setenv("APPDATA", str(tmp_path / "empty-roaming"))
    monkeypatch.setenv("HOME", str(tmp_path / "empty-home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "empty-home"))
    resolved = get_amverge_config_dir()
    assert resolved.endswith("com.amverge.cli")


def test_app_paths_json_supplies_config_and_hf_storage(monkeypatch, tmp_path):
    app_data = tmp_path / "roaming" / "app.amverge"
    app_data.mkdir(parents=True)
    config = tmp_path / "moved" / "config"
    hf_home = tmp_path / "moved" / "hf-cache"
    (app_data / "paths.json").write_text(json.dumps({
        "ai_config_dir": str(config),
        "hf_home_dir": str(hf_home),
        "scene_scout_dir": str(tmp_path / "moved" / "amverge-scene-scout"),
    }), encoding="utf-8")
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    monkeypatch.delenv("AMVERGE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)
    monkeypatch.delenv("AMVERGE_SCENE_SCOUT_DIR", raising=False)

    assert get_amverge_config_dir() == str(config)
    configure_app_storage()
    assert os.environ["HF_HOME"] == str(hf_home)
    assert os.environ["AMVERGE_SCENE_SCOUT_DIR"] == str(tmp_path / "moved" / "amverge-scene-scout")


def test_legacy_ai_pointer_supplies_config_and_hf_storage(monkeypatch, tmp_path):
    app_data = tmp_path / "roaming" / "app.amverge"
    app_data.mkdir(parents=True)
    root = tmp_path / "custom-ai"
    (app_data / "ai_storage_path.txt").write_text(str(root), encoding="utf-8")
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    monkeypatch.delenv("AMVERGE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)

    assert get_amverge_config_dir() == str(root / "config")
    configure_app_storage()
    assert os.environ["HF_HOME"] == str(root / "hf-cache")


def test_null_json_fields_do_not_mask_legacy_pointer(monkeypatch, tmp_path):
    app_data = tmp_path / "roaming" / "app.amverge"
    app_data.mkdir(parents=True)
    root = tmp_path / "legacy-ai"
    (app_data / "paths.json").write_text(json.dumps({
        "ai_config_dir": None,
        "hf_home_dir": None,
    }), encoding="utf-8")
    (app_data / "ai_storage_path.txt").write_text(str(root), encoding="utf-8")
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    monkeypatch.delenv("AMVERGE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_CACHE", raising=False)

    assert get_amverge_config_dir() == str(root / "config")
    configure_app_storage()
    assert os.environ["HF_HOME"] == str(root / "hf-cache")


def test_explicit_storage_env_wins_over_app_paths(monkeypatch, tmp_path):
    app_data = tmp_path / "roaming" / "app.amverge"
    app_data.mkdir(parents=True)
    (app_data / "paths.json").write_text(json.dumps({
        "ai_config_dir": str(tmp_path / "app-config"),
        "hf_home_dir": str(tmp_path / "app-hf"),
    }), encoding="utf-8")
    config = tmp_path / "explicit-config"
    hf_home = tmp_path / "explicit-hf"
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    monkeypatch.setenv("AMVERGE_CONFIG_DIR", str(config))
    monkeypatch.setenv("HF_HOME", str(hf_home))

    configure_app_storage()
    assert get_amverge_config_dir() == str(config)
    assert os.environ["HF_HOME"] == str(hf_home)


def test_malformed_app_pointer_falls_back_to_default(monkeypatch, tmp_path):
    app_data = tmp_path / "roaming" / "app.amverge"
    app_data.mkdir(parents=True)
    (app_data / "paths.json").write_text("not json", encoding="utf-8")
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    monkeypatch.setenv("HOME", str(tmp_path / "empty-home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "empty-home"))
    monkeypatch.delenv("AMVERGE_CONFIG_DIR", raising=False)

    assert get_amverge_config_dir().endswith("com.amverge.cli")
