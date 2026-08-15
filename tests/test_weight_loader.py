from __future__ import annotations

import os
from collections import OrderedDict

from amverge.core.interpolation.weight_loader import (
    _extract_rife_state_dict,
    get_weight_path,
    _get_flow_weights_path,
    _get_weights_dir,
)


class TestExtractRIFEDict:
    def test_plain_state_dict(self):
        sd = {"block0.conv0.weight": 1, "block1.conv0.weight": 2}
        result = _extract_rife_state_dict(sd)
        assert "block0.conv0.weight" in result
        assert "block1.conv0.weight" in result

    def test_strip_module_prefix(self):
        sd = {"module.block0.conv0.weight": 1, "module.flownet.block1.conv0.weight": 2}
        result = _extract_rife_state_dict(sd)
        assert "block0.conv0.weight" in result
        assert "block1.conv0.weight" in result

    def test_strip_flownet_prefix(self):
        sd = {"flownet.block0.conv0.weight": 1, "block1.conv0.weight": 2}
        result = _extract_rife_state_dict(sd)
        assert "block0.conv0.weight" in result
        assert "block1.conv0.weight" in result

    def test_double_strip(self):
        sd = {"module.flownet.encode.cnn0.weight": 1}
        result = _extract_rife_state_dict(sd)
        assert "encode.cnn0.weight" in result

    def test_non_dict_input(self):
        result = _extract_rife_state_dict(42)
        assert isinstance(result, OrderedDict)
        assert len(result) == 0

    def test_ema_key(self):
        sd = {"ema_model_state_dict": {"block0.conv0.weight": 1}}
        result = _extract_rife_state_dict(sd)
        assert "block0.conv0.weight" in result


class TestWeightPaths:
    def test_weights_dir_exists(self):
        path = _get_weights_dir()
        assert path.endswith("interpolation")

    def test_get_weight_path(self):
        path = get_weight_path("rife4.25")
        assert path.endswith("rife425.pth")
        assert "interpolation" in path

    def test_get_flow_weights_path(self):
        path = _get_flow_weights_path("gmflow_sintel-0c07dcb3.pth")
        assert "flow" in path
        assert path.endswith("gmflow_sintel-0c07dcb3.pth")
