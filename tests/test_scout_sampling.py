import subprocess
import sys
import textwrap

import av
import numpy as np
import pytest

from amverge.core.scenescout import indexing

FPS = 24
GOP = 24
WIDTH, HEIGHT = 320, 180
BLANK_FRAMES = range(48, 60)
TOTAL_FRAMES = 120


def textured(index: int) -> np.ndarray:
    rng = np.random.default_rng(0)
    image = rng.integers(40, 216, (HEIGHT // 4, WIDTH // 4, 3), dtype=np.uint8).repeat(4, axis=0).repeat(4, axis=1)
    for bit in range(8):
        image[:16, bit * 16 : (bit + 1) * 16] = 255 if index >> bit & 1 else 0
    return image


@pytest.fixture(scope="module")
def video(tmp_path_factory):
    path = tmp_path_factory.mktemp("sampling") / "video.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = WIDTH, HEIGHT, "yuv420p"
        stream.options = {"crf": "10", "x264-params": f"keyint={GOP}:min-keyint={GOP}:scenecut=0"}
        for index in range(TOTAL_FRAMES):
            pixels = np.full((HEIGHT, WIDTH, 3), 255, np.uint8) if index in BLANK_FRAMES else textured(index)
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def frame_number(frame: np.ndarray) -> int:
    return sum(1 << bit for bit in range(8) if frame[4:12, bit * 16 + 4 : bit * 16 + 12].mean() > 128)


def ms(frame: int) -> int:
    return frame * 1000 // FPS + 1


def test_first_frame_on_and_off_a_keyframe(video):
    scenes = [(ms(0), ms(24)), (ms(24), ms(36)), (ms(36), ms(48)), (ms(72), ms(96))]
    frames = indexing.sample_scene_frames(video, scenes)
    assert [frame_number(f) for f in frames] == [0, 24, 37, 72]


def test_blank_first_frame_falls_back_to_the_midpoint(video):
    frames = indexing.sample_scene_frames(video, [(ms(48), ms(72))])
    assert frame_number(frames[0]) == 61


def test_matches_the_full_decode_exactly(video):
    scenes = [(ms(i), ms(i + 7)) for i in range(0, 48, 7)]
    frames = indexing.sample_scene_frames(video, scenes)
    targets = indexing._scene_targets(scenes, *indexing._packet_times(video))
    truth = indexing.sample_frames(video, [(t, t) for t in targets])
    assert len(frames) == len(scenes)
    for got, want in zip(frames, truth):
        assert np.array_equal(got, want)


def test_scene_past_the_end_still_gets_a_frame(video):
    scenes = [(ms(0), ms(24)), (ms(500), ms(600))]
    frames = indexing.sample_scene_frames(video, scenes)
    assert len(frames) == 2
    assert frame_number(frames[1]) >= TOTAL_FRAMES - GOP


def test_packets_are_skipped_between_keyframe_scenes(video, monkeypatch):
    converted = []
    real_blank = indexing.is_blank_frame
    monkeypatch.setattr(indexing, "is_blank_frame", lambda frame: converted.append(1) or real_blank(frame))
    scenes = [(ms(0), ms(24)), (ms(24), ms(48)), (ms(72), ms(96)), (ms(96), ms(119))]
    frames = indexing.sample_scene_frames(video, scenes)
    assert [frame_number(f) for f in frames] == [0, 24, 72, 96]
    assert len(converted) == 4


def test_snap_prefers_a_keyframe_within_two_frames():
    keyframes = [0, 1000, 2000]
    times = [0, 500, 1000, 1125, 2000, 2500]
    scenes = [(958, 1500), (1100, 1500), (2083, 2500)]
    assert indexing._scene_targets(scenes, keyframes, times, 24.0) == [1000, 1125, 2000]


def test_blank_check_tells_empty_frames_from_detail():
    assert indexing.is_blank_frame(np.full((180, 320, 3), 255, np.uint8))
    assert indexing.is_blank_frame(np.zeros((180, 320, 3), np.uint8))
    two_tone = np.zeros((180, 320, 3), np.uint8)
    two_tone[:, :160] = 200
    assert indexing.is_blank_frame(two_tone)
    assert not indexing.is_blank_frame(textured(0))


def test_embedder_returns_vectors_in_scene_order(monkeypatch):
    import amverge.core.scenescout.embedding as embedding

    monkeypatch.setattr(embedding, "embed_frames", lambda batch, **_: np.array([[f] for f in batch], np.float32))
    embedder = indexing._BackgroundEmbedder(device=None, max_patches=256, batch_size=2)
    for index in (0, 2, 3, 1):
        embedder.add(index, float(index))
    vectors = embedder.finish(4, lambda *_: None)
    assert vectors[:, 0].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_close_video_is_safe_after_stopping_early(video):
    script = textwrap.dedent(
        f"""
        import av
        from amverge.core.scenescout.decode import close_video

        def run():
            container = av.open({str(video)!r})
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for index, _ in enumerate(container.decode(stream)):
                if index >= 12:
                    break
            close_video(container, stream)

        run()
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True, timeout=60)
