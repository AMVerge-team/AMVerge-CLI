import os
from amverge.core.infra.config import get_amverge_config_dir, get_ffmpeg_dir, get_models_dir
from amverge.core.depth.registry import _get_depth_models_dir


def test_config_dir_env_override(monkeypatch, tmp_path):
    custom = tmp_path / "amverge-ai" / "config"
    monkeypatch.setenv("AMVERGE_CONFIG_DIR", str(custom))

    assert get_amverge_config_dir() == str(custom)
    assert get_models_dir() == os.path.join(str(custom), "models", "upscale")
    assert get_ffmpeg_dir() == os.path.join(str(custom), "ffmpeg")
    assert _get_depth_models_dir() == os.path.join(str(custom), "models", "depth")


def test_config_dir_falls_back_when_unset(monkeypatch):
    monkeypatch.delenv("AMVERGE_CONFIG_DIR", raising=False)
    resolved = get_amverge_config_dir()
    assert resolved.endswith("com.amverge.cli")
