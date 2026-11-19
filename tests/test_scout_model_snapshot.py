import pytest

from amverge.core.scenescout.embedding import (
    SceneScoutModelUnavailable,
    _model_snapshot,
)


def test_complete_snapshot_loads_offline(monkeypatch, tmp_path):
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    assert _model_snapshot(None) == str(tmp_path)
    assert calls[0]["local_files_only"] is True
    assert calls[0]["allow_patterns"] == ["*.json", "*.safetensors", "*.model", "*.jinja"]


def test_incomplete_snapshot_is_repaired_online(monkeypatch, tmp_path):
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        if kwargs.get("local_files_only"):
            raise OSError("missing weights")
        return str(tmp_path)

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    assert _model_snapshot("token") == str(tmp_path)
    assert len(calls) == 2
    assert calls[1]["local_files_only"] is False


def test_unavailable_snapshot_has_recovery_message(monkeypatch):
    def download(**kwargs):
        raise OSError("offline")

    monkeypatch.setattr("huggingface_hub.snapshot_download", download)
    with pytest.raises(SceneScoutModelUnavailable, match="model cache is incomplete"):
        _model_snapshot(None)
