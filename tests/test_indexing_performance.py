from __future__ import annotations

from unittest import mock

import numpy as np


def test_windowed_scorer_push_batch_matches_push():
    from amverge.core.detection.ai_scene_detection import _WindowedScorer

    class _DummyModel:
        def __call__(self, tensor):
            scores = tensor.float().mean(dim=(2, 3, 4)).unsqueeze(-1) / 255.0
            return scores, None

    rng = np.random.default_rng(42)
    frames = rng.integers(0, 256, size=(283, 27, 48, 3), dtype=np.uint8)

    s_single = _WindowedScorer(_DummyModel(), "cpu", window_batch=4)
    for frame in frames:
        s_single.push(frame)
    p_single = s_single.finish()

    s_batch = _WindowedScorer(_DummyModel(), "cpu", window_batch=4)
    for i in range(0, len(frames), 64):
        s_batch.push_batch(frames[i : i + 64])
    p_batch = s_batch.finish()

    assert p_single.shape == (283,)
    assert np.array_equal(p_single, p_batch)


def test_ffmpeg_hwaccel_fallback_to_cpu(tmp_path):
    from amverge.core.detection.ai_scene_detection import _open_ffmpeg_frame_stream
    from amverge.core.transnet.transnet_constants import FRAME_BYTES

    fake_frame = b"\x11" * FRAME_BYTES

    class _FakeStdout:
        def __init__(self, data: bytes):
            self._data = data
            self._pos = 0

        def read(self, n: int) -> bytes:
            chunk = self._data[self._pos : self._pos + n]
            self._pos += len(chunk)
            return chunk

        def close(self) -> None:
            pass

    class _FakeProc:
        def __init__(self, data: bytes):
            self.stdout = _FakeStdout(data)

        def terminate(self) -> None:
            pass

        def kill(self) -> None:
            pass

        def wait(self) -> int:
            return 0

    calls: list[list[str]] = []

    def _fake_popen(cmd, **kwargs):
        calls.append(cmd)
        return _FakeProc(b"" if len(calls) == 1 else fake_frame)

    with mock.patch("amverge.core.detection.ai_scene_detection.subprocess.Popen", side_effect=_fake_popen), \
         mock.patch("amverge.core.detection.ai_scene_detection.sys.platform", "win32"):
        label, proc, first_chunk = _open_ffmpeg_frame_stream(tmp_path / "ep.mkv", "hevc", "cuda", FRAME_BYTES * 4)

    assert len(calls) == 2
    assert label == "cpu"
    assert first_chunk == fake_frame
    proc.wait()


def test_merge_short_scenes_in_memory():
    from amverge.core.detection.short_scenes import merge_short_scenes

    fps = 24.0
    frames = np.zeros((48, 27, 48, 3), dtype=np.uint8)
    frames[:18, :, :, 0] = 200
    frames[18:22, :, :, 2] = 220
    frames[22:, :, :, 2] = 210

    scenes = [
        {"scene_index": 0, "start_sec": 0.0, "end_sec": 0.75, "duration_sec": 0.75},
        {"scene_index": 1, "start_sec": 0.75, "end_sec": 0.916, "duration_sec": 0.166},
        {"scene_index": 2, "start_sec": 0.916, "end_sec": 2.0, "duration_sec": 1.084},
    ]
    merged = merge_short_scenes(scenes, "nonexistent.mkv", decoded_frames=frames, fps=fps)
    assert len(merged) == 2
    assert merged[0]["end_sec"] == 0.75
    assert merged[1]["start_sec"] == 0.75
    assert [s["scene_index"] for s in merged] == [0, 1]


def test_decode_skips_discrete_hwaccel_for_10bit():
    from amverge.core.scenescout import decode

    with mock.patch.object(decode, "_hw_device", return_value="cuda"), \
         mock.patch.object(decode, "_is_high_bit_depth", return_value=True), \
         mock.patch("av.open", return_value="cpu_container") as mock_open:
        res = decode.open_video("sample.mkv")

    assert res == "cpu_container"
    assert mock_open.call_count == 1
    assert "hwaccel" not in mock_open.call_args.kwargs
