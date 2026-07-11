from __future__ import annotations

import json
from pathlib import Path

from amverge.core.interpolation.registry import (
    get_model,
    get_rife_models,
    get_pervfi_models,
    get_all_model_keys,
    INTERPOLATION_REGISTRY,
    QUALITY_PRESETS,
)


class TestRegistryLoad:
    def test_registry_not_empty(self):
        assert len(INTERPOLATION_REGISTRY) > 0

    def test_source_present(self):
        json_path = Path(__file__).parent.parent / "amverge" / "core" / "interpolation" / "registry.json"
        data = json.loads(json_path.read_text())
        assert "_source" in data
        assert "interpolation" in data["_source"]


class TestPerVFIModels:
    def test_pervfi_model_exists(self):
        model = get_model("pervfi")
        assert model is not None
        assert model["method"] == "pervfi"
        assert model["generator"] == "v00"
        assert "flow_file" in model

    def test_pervfi_vb_model_exists(self):
        model = get_model("pervfi-vb")
        assert model is not None
        assert model["method"] == "pervfi"
        assert model["generator"] == "vb"

    def test_get_pervfi_models(self):
        models = get_pervfi_models()
        assert "pervfi" in models
        assert "pervfi-vb" in models
        assert all(v["method"] == "pervfi" for v in models.values())

    def test_pervfi_has_flow_estimator(self):
        model = get_model("pervfi")
        assert model["flow_estimator"] == "gmflow"
        assert model["flow_file"] == "gmflow_sintel-0c07dcb3.pth"


class TestRIFEModels:
    def test_get_rife_models(self):
        models = get_rife_models()
        assert len(models) > 5
        assert all(v["method"] == "rife" for v in models.values())

    def test_rife_does_not_include_pervfi(self):
        rife = get_rife_models()
        assert "pervfi" not in rife
        assert "pervfi-vb" not in rife


class TestModelLookup:
    def test_get_known_model(self):
        model = get_model("rife4.25")
        assert model is not None
        assert model["method"] == "rife"

    def test_get_unknown_model(self):
        model = get_model("nonexistent_model_key")
        assert model is None

    def test_all_keys_include_pervfi(self):
        keys = get_all_model_keys()
        assert "pervfi" in keys
        assert "pervfi-vb" in keys


class TestQualityPresets:
    def test_presets_exist(self):
        assert "archival" in QUALITY_PRESETS
        assert "high" in QUALITY_PRESETS
        assert "balanced" in QUALITY_PRESETS
        assert "fast" in QUALITY_PRESETS
        assert "draft" in QUALITY_PRESETS

    def test_preset_has_required_keys(self):
        for name, preset in QUALITY_PRESETS.items():
            assert "crf" in preset
            assert "x264" in preset
            assert "tune" in preset
