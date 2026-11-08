from __future__ import annotations

import json

from typer.testing import CliRunner

from amverge.cli import app
from amverge.core.upscaling.registry import UPSCALE_REGISTRY

runner = CliRunner()


def _json_output(result):
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_models_json_lists_every_upscale_model():
    payload = _json_output(runner.invoke(app, ["models", "--json"]))
    assert {m["key"] for m in payload["upscale"]} == set(UPSCALE_REGISTRY)


def test_ml_models_state_their_download_size():
    payload = _json_output(runner.invoke(app, ["models", "--json", "--upscale"]))
    ml = [m for m in payload["upscale"] if m["method"] == "ml"]
    assert ml and all(m["sizeBytes"] > 0 for m in ml)


def test_models_json_upscale_filter_leaves_out_the_others():
    payload = _json_output(runner.invoke(app, ["models", "--json", "--upscale"]))
    assert payload["depth"] == [] and payload["interpolation"] == []


def test_deleting_an_unknown_model_reports_failure():
    payload = _json_output(runner.invoke(app, ["models", "--json", "--delete", "nope"]))
    assert payload["result"]["ok"] is False


def test_upscale_ipc_rejects_an_unknown_model(tmp_path):
    source = tmp_path / "in.mp4"
    source.write_bytes(b"x")
    result = runner.invoke(app, ["upscale", str(source), "--output", str(tmp_path / "o.mp4"), "--ipc", "--model", "nope"])
    assert result.exit_code == 1
