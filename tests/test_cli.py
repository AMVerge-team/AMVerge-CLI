from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from amverge.cli import app
from tests.fixtures import BY_NAME, media_params
from tests.fixtures.media import audio_streams, read_frames, video_stream

pytestmark = pytest.mark.media

runner = CliRunner()


def invoke(*args: str):
    return runner.invoke(app, [str(a) for a in args], catch_exceptions=False)


def numbers(path) -> list[int]:
    return [f.number for f in read_frames(path)]


def last_json_line(text: str) -> dict:
    for line in reversed(text.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise AssertionError(f"no JSON line in output:\n{text[-2000:]}")


def manifest(tmp_path: Path, spec, keys=("start_sec", "end_sec")) -> Path:
    path = tmp_path / "scenes.json"
    path.write_text(json.dumps({"scenes": [
        {"scene_index": i, keys[0]: s, keys[1]: e}
        for i, (s, e) in enumerate(spec.scene_secs)
    ]}))
    return path


class TestExportCommand:
    @pytest.mark.parametrize("args,message", [
        (["--container", "mkv"], "isn't supported yet"),
        (["--codec", "prores_422_hq", "--container", "mp4"], "not compatible"),
        (["--codec", "vp9"], "Unknown codec"),
        (["--audio", "wav"], "Unknown audio"),
        (["--hardware", "tpu"], "Unknown hardware"),
    ])
    def test_rejects_invalid_settings(self, media_files, tmp_path, args, message):
        spec = BY_NAME["h264_24_cuts"]
        result = invoke("export", media_files[spec.name], "--scenes", manifest(tmp_path, spec),
                        "-o", tmp_path / "out", *args)
        assert result.exit_code == 1
        assert message in result.output

    def test_select_and_reencode(self, media_files, tmp_path):
        spec = BY_NAME["h265_25_cuts"]
        out = tmp_path / "out"
        result = invoke("export", media_files[spec.name], "--scenes", manifest(tmp_path, spec),
                        "-o", out, "--select", "1-2", "--codec", "h264", "--hardware", "cpu", "--name", "ep")
        assert result.exit_code == 0, result.output
        files = sorted(out.iterdir())
        assert [f.name for f in files] == ["ep_0001.mp4", "ep_0002.mp4"]
        for f, idx in zip(files, [1, 2]):
            assert numbers(f) == list(range(*spec.scene_frames[idx]))
            assert video_stream(f)["profile"] == "Main"

    def test_merge_prores_to_mov(self, media_files, tmp_path):
        spec = BY_NAME["h264_24_cuts"]
        out = tmp_path / "out"
        result = invoke("export", media_files[spec.name], "--scenes", manifest(tmp_path, spec),
                        "-o", out, "--select", "0,2,4", "--merge", "--codec", "prores_422_hq",
                        "--container", "mov", "--audio", "pcm24")
        assert result.exit_code == 0, result.output
        merged = out / f"{media_files[spec.name].stem}.mov"
        assert numbers(merged) == [n for i in (0, 2, 4) for n in range(*spec.scene_frames[i])]
        assert audio_streams(merged)[0]["codec_name"] == "pcm_s24le"

    def test_pcm_copy_into_mp4_falls_back_to_aac(self, media_files, tmp_path):
        spec = BY_NAME["prores_23976_422hq"]
        out = tmp_path / "out"
        result = invoke("export", media_files[spec.name], "--scenes", manifest(tmp_path, spec),
                        "-o", out, "--select", "1", "--codec", "h264_high", "--container", "mp4")
        assert result.exit_code == 0, result.output
        clip = next(out.iterdir())
        assert audio_streams(clip)[0]["codec_name"] == "aac"

    def test_prores_copy_export_to_mov(self, media_files, tmp_path):
        spec = BY_NAME["prores_50_4444"]
        out = tmp_path / "out"
        result = invoke("export", media_files[spec.name], "--scenes", manifest(tmp_path, spec),
                        "-o", out, "--select", "1", "--codec", "copy", "--container", "mov")
        assert result.exit_code == 0, result.output
        clip = next(out.iterdir())
        assert clip.suffix == ".mov"
        assert video_stream(clip)["codec_name"] == "prores"
        assert numbers(clip) == list(range(*spec.scene_frames[1]))

    def test_ipc_mode_prints_outputs_json(self, media_files, tmp_path):
        spec = BY_NAME["h264_24_cuts"]
        result = invoke("export", media_files[spec.name], "--scenes", manifest(tmp_path, spec),
                        "-o", tmp_path / "out", "--select", "3", "--codec", "h264_main",
                        "--hardware", "cpu", "--ipc")
        assert result.exit_code == 0
        payload = last_json_line(result.stdout)
        assert payload["error"] is None
        assert len(payload["outputs"]) == 1 and os.path.exists(payload["outputs"][0])

    def test_inputs_json_app_mode(self, media_files, tmp_path):
        spec = BY_NAME["h264_24_cuts"]
        items = [
            {"input": str(media_files[spec.name]), "scene_index": i, "start_sec": s, "end_sec": e}
            for i, (s, e) in enumerate(spec.scene_secs) if i in (1, 3)
        ]
        inputs = tmp_path / "inputs.json"
        inputs.write_text(json.dumps(items))
        out = tmp_path / "out"
        result = invoke("export", "--inputs-json", inputs, "-o", out, "--codec", "h264_main",
                        "--hardware", "cpu", "--merge")
        assert result.exit_code == 0, result.output
        merged = next(out.iterdir())
        assert numbers(merged) == [n for i in (1, 3) for n in range(*spec.scene_frames[i])]


class TestDetectThenExport:
    def test_detect_json_feeds_export(self, media_files, tmp_path):
        spec = BY_NAME["h264_24_cuts"]
        detected = tmp_path / "detected.json"
        result = invoke("detect", media_files[spec.name], "-o", tmp_path / "scenes",
                        "--json-output", detected, "--no-similarity", "--no-rpc")
        assert result.exit_code == 0, result.output
        scenes = json.loads(detected.read_text())["scenes"]
        assert len(scenes) == len(spec.segments)

        out = tmp_path / "out"
        result = invoke("export", media_files[spec.name], "--scenes", detected, "-o", out,
                        "--select", "1", "--codec", "h264_main", "--hardware", "cpu")
        assert result.exit_code == 0, result.output
        assert numbers(next(out.iterdir())) == list(range(*spec.scene_frames[1]))

    def test_keyframes_command_json(self, media_files):
        spec = BY_NAME["h264_2997_gop"]
        result = invoke("keyframes", media_files[spec.name], "--json")
        assert result.exit_code == 0
        assert json.loads(result.stdout.strip().splitlines()[-1]) == pytest.approx(spec.keyframe_secs, abs=1e-4)


class TestBackendSidecar:
    @pytest.mark.parametrize("spec", media_params("h264_24_cuts", "h265_25_cuts", "h264_2997_gop"))
    def test_keyframe_import_then_copy_merge(self, media_files, tmp_path, spec):
        work = tmp_path / "work"
        result = invoke("backend", media_files[spec.name], work, "keyframe_detection", "video_files")
        assert result.exit_code == 0, result.output
        payload = last_json_line(result.stdout)
        assert payload["error"] is None
        scenes = payload["scenes"]
        assert [s["scene_index"] for s in scenes] == list(range(len(scenes)))
        assert scenes[0]["start_sec"] == 0.0
        for prev, cur in zip(scenes, scenes[1:]):
            assert cur["start_sec"] == pytest.approx(prev["end_sec"])
        clip_numbers = [n for s in scenes for n in numbers(s["clip_path"])]
        assert clip_numbers == list(range(spec.total_frames))
        assert all(s["thumbnail_ready"] for s in scenes)

        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(payload))
        out = tmp_path / "out"
        result = invoke("export", media_files[spec.name], "--scenes", manifest_path, "-o", out,
                        "--merge", "--codec", "copy")
        assert result.exit_code == 0, result.output
        assert numbers(next(out.iterdir())) == list(range(spec.total_frames))

    @pytest.mark.ml
    @pytest.mark.slow
    @pytest.mark.parametrize("spec", media_params("h264_24_cuts", "h264_2997_gop"))
    def test_transnet_import_gives_exact_previews(self, media_files, tmp_path, spec, transnet):
        work = tmp_path / "work"
        result = invoke("backend", media_files[spec.name], work, "transnetv2_cpu", "video_files")
        assert result.exit_code == 0, result.output
        payload = last_json_line(result.stdout)
        scenes = payload["scenes"]
        assert [spec.frame_at(s["start_sec"]) for s in scenes] == [0, *spec.cut_frames]
        for i, s in enumerate(scenes):
            assert s["clip_mode"] == "reencode"
            assert numbers(s["clip_path"]) == list(range(*spec.scene_frames[i])), f"scene {i}"


@pytest.mark.ml
@pytest.mark.slow
def test_backend_repairs_stale_transnet_cache(media_files, tmp_path, transnet):
    import numpy as np
    from amverge.core.infra.ipc import build_video_cache_prefix

    spec = BY_NAME["h265_60_main10_gop"]
    video = media_files[spec.name]
    work = tmp_path / "work"
    work.mkdir()
    prefix = build_video_cache_prefix(video)
    inclusive = np.array([[a, b - 1] for a, b in spec.scene_frames])
    np.save(work / f"{prefix}_transnet_frames.npy", inclusive)
    np.save(work / f"{prefix}_transnet_secs.npy", np.round(inclusive / 60.0, 2))

    result = invoke("backend", video, work, "transnetv2_cpu", "video_files")
    assert result.exit_code == 0, result.output
    payload = last_json_line(result.stdout)
    assert payload["cache"]["cache_hit"]
    scenes = payload["scenes"]
    assert [(s["start_sec"], s["end_sec"]) for s in scenes] == pytest.approx(spec.scene_secs, abs=1e-6)
    for i, s in enumerate(scenes):
        assert numbers(s["clip_path"]) == list(range(*spec.scene_frames[i])), f"scene {i}"
