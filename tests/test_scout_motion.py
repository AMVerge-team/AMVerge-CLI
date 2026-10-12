"""Scene Scout motion tags: measurement on synthetic footage with known motion,
the tag/flow rules, and the database + search integration (no model needed)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from amverge.core.scenescout import db as scoutdb
from amverge.core.scenescout import motion
from amverge.core.scenescout.search import search
from amverge.core.scenescout.types import SearchOptions

W, H, FPS = 480, 270, 24.0


def _texture(height, width, seed):
    rng = np.random.default_rng(seed)
    noise = rng.integers(0, 255, (height, width), dtype=np.uint8)
    return cv2.GaussianBlur(noise, (0, 0), 2.0)


def _times(n):
    return [i / FPS for i in range(n)]


def _pan(dx_per_frame, n=10):
    """Camera pan: a big texture slides under a fixed window."""
    world = _texture(H + 40, W + 400, seed=1)
    x0 = 200
    return [world[20 : 20 + H, x0 + int(round(i * dx_per_frame)) : x0 + int(round(i * dx_per_frame)) + W] for i in range(n)]


def _subject(dx_per_frame, dy_per_frame=0.0, n=10):
    """A textured square moving over a still textured background."""
    background = _texture(H, W, seed=2)
    patch = _texture(80, 80, seed=3)
    frames = []
    for i in range(n):
        frame = background.copy()
        x, y = int(round(150 + i * dx_per_frame)), int(round(95 + i * dy_per_frame))
        frame[y : y + 80, x : x + 80] = patch
        frames.append(frame)
    return frames


def _zoom(scale_per_frame, n=10):
    base = _texture(H, W, seed=4)
    centre = (W / 2, H / 2)
    return [cv2.warpAffine(base, cv2.getRotationMatrix2D(centre, 0, scale_per_frame**i), (W, H)) for i in range(n)]


# ---------------------------------------------------------------- measurement


def test_content_moving_left_reads_as_camera_panning_right():
    frames = _pan(dx_per_frame=6)
    tags = motion.tags_for(motion.measure_boundary(frames, _times(len(frames))))
    assert tags["camera"] == "pan-right"
    assert tags["subject"] == "still"


def test_subject_moving_right_over_a_still_background():
    frames = _subject(dx_per_frame=8)
    tags = motion.tags_for(motion.measure_boundary(frames, _times(len(frames))))
    assert tags["camera"] == "static"
    assert tags["subject"] == "right"
    assert tags["speed"] in motion.SPEED_TAGS


def test_subject_moving_up_left():
    frames = _subject(dx_per_frame=-6, dy_per_frame=-6)
    tags = motion.tags_for(motion.measure_boundary(frames, _times(len(frames))))
    assert tags["subject"] == "up-left"


def test_zoom_in():
    frames = _zoom(1.03)
    assert motion.tags_for(motion.measure_boundary(frames, _times(len(frames))))["camera"] == "zoom-in"


def test_static_shot_is_still_not_unclear():
    frames = [_texture(H, W, seed=5)] * 8
    tags = motion.tags_for(motion.measure_boundary(frames, _times(len(frames))))
    assert tags == {"subject": "still", "camera": "static", "speed": None}


def test_featureless_or_short_input_is_unclear():
    flat = [np.full((H, W), 128, np.uint8)] * 8
    assert motion.measure_boundary(flat, _times(8)) is None
    assert motion.tags_for(None) == {"subject": "unclear", "camera": "unclear", "speed": None}
    assert motion.measure_boundary(_subject(8, n=2), _times(2)) is None


# ---------------------------------------------------------------- tag rules


@pytest.mark.parametrize(
    "vx,vy,expected",
    [(1, 0, "right"), (1, 1, "down-right"), (0, 1, "down"), (-1, 0, "left"), (0, -1, "up"), (1, -1, "up-right")],
)
def test_direction8_uses_screen_y_down(vx, vy, expected):
    assert motion.direction8(vx, vy) == expected


# ---------------------------------------------------------------- db + search


def _unit(seed, dim=8):
    v = np.random.default_rng(seed).normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _raw_moving(vx, vy):
    """A raw boundary that tags as a subject moving with velocity (vx, vy)."""
    return {"v": 1, "pairs": 6, "cam": [0.0, 0.0, 0.0], "camConf": 0.9, "subjPairs": 6,
            "subjSupport": 40, "subj": [vx, vy], "subjRel": [vx, vy], "subjConsistency": 0.95}


def _build_db(root):
    path = scoutdb.create_database("flow", root=root)
    video_id = scoutdb.upsert_video(path, str(root / "ep.mkv"), 0.0)
    query = _unit(0)
    # scene 0 most similar but moves left; 1 moves right; 2 moves down-right; 3 has no motion row
    vectors = [query, _unit(1) * 0.5 + query * 0.5, _unit(2), _unit(3)]
    rows = [(i, i * 1000, i * 1000 + 900, scoutdb_pack(v / np.linalg.norm(v)), None) for i, v in enumerate(vectors)]
    scoutdb.insert_scenes(path, video_id, rows)
    scoutdb.insert_scene_motion(
        path,
        video_id,
        {0: (_raw_moving(-0.5, 0), None), 1: (_raw_moving(0.5, 0), None), 2: (_raw_moving(0.4, 0.4), None)},
        motion.MOTION_VERSION,
    )
    scoutdb.mark_video_complete(path, video_id)
    return path, query


def scoutdb_pack(vector):
    return np.asarray(vector, dtype=np.float32).tobytes()


def test_motion_round_trips_and_tags_every_hit():
    with tempfile.TemporaryDirectory() as td:
        path, query = _build_db(Path(td))
        hits = search(query, SearchOptions(databases=[str(path)], include_thumbnails=False))
        by_index = {h.scene_index: h for h in hits}
        assert by_index[0].motion["start"]["subject"] == "left"
        assert by_index[1].motion["start"]["subject"] == "right"
        assert by_index[3].motion is None
        assert hits[0].scene_index == 0  # no motion options: plain similarity order


def test_start_subject_filter():
    with tempfile.TemporaryDirectory() as td:
        path, query = _build_db(Path(td))
        options = SearchOptions(databases=[str(path)], include_thumbnails=False, start_subjects=["right"])
        assert [h.scene_index for h in search(query, options)] == [1]
        options.include_unclear = True
        assert sorted(h.scene_index for h in search(query, options)) == [1, 3]


def test_motion_never_reorders_or_drops_results_without_a_filter():
    # a flow search is a plain similarity search: the closest scene comes first
    # whichever way it moves, and scenes without motion are still returned
    with tempfile.TemporaryDirectory() as td:
        path, query = _build_db(Path(td))
        hits = search(query, SearchOptions(databases=[str(path)], include_thumbnails=False))
        assert len(hits) == 4
        assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
        assert hits[0].scene_index == 0 and hits[0].motion["start"]["subject"] == "left"


def test_exclude_range_drops_the_source_scene_itself():
    with tempfile.TemporaryDirectory() as td:
        path, query = _build_db(Path(td))
        options = SearchOptions(databases=[str(path)], include_thumbnails=False,
                                exclude_range=(str(Path(td) / "ep.mkv"), 0, 900))
        assert 0 not in [h.scene_index for h in search(query, options)]


def test_backfill_finds_videos_missing_motion():
    with tempfile.TemporaryDirectory() as td:
        path, _ = _build_db(Path(td))
        pending = scoutdb.videos_missing_motion(path, motion.MOTION_VERSION)
        assert len(pending) == 1 and len(pending[0][2]) == 4  # scene 3 has no row, so the video is redone


def test_deleting_a_video_removes_its_motion():
    with tempfile.TemporaryDirectory() as td:
        path, _ = _build_db(Path(td))
        assert scoutdb.remove_video(path, filepath=str(Path(td) / "ep.mkv"))
        with scoutdb.connect(path) as conn:
            assert conn.execute("SELECT COUNT(*) FROM scene_motion").fetchone()[0] == 0


# ---------------------------------------------------------------- frame conversion


@pytest.mark.parametrize("pix_fmt", ["yuv420p", "yuvj420p", "yuv420p10le", "rgb24"])
def test_frame_to_grey_matches_pyav_conversion(pix_fmt):
    av = pytest.importorskip("av")
    rgb = cv2.cvtColor(_texture(H, W, seed=7), cv2.COLOR_GRAY2RGB)
    frame = av.VideoFrame.from_ndarray(rgb, format="rgb24").reformat(format=pix_fmt)
    fast = motion.frame_to_grey(frame, 160, 90).astype(int)
    reference = frame.reformat(width=160, height=90, format="gray").to_ndarray().astype(int)
    assert fast.shape == (90, 160)
    assert np.abs(fast - reference).mean() < 3.0


# ---------------------------------------------------------------- one real decode


def test_collect_scene_motion_over_an_encoded_video(tmp_path):
    video = tmp_path / "synthetic.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        pytest.skip("no mp4 encoder in this OpenCV build")
    # scene A: subject moves right for 1s; scene B: camera pans for 1s
    for frame in _subject(dx_per_frame=6, n=24) + _pan(dx_per_frame=6, n=24):
        writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
    writer.release()

    results = motion.collect_scene_motion(str(video), [(0, 1000), (1000, 2000)])
    assert set(results) == {0, 1}
    assert motion.tags_for(results[0][1])["subject"] == "right"
    assert motion.tags_for(results[1][0])["camera"] == "pan-right"

    clip = motion.measure_clip(str(video), 0, 1000)
    assert motion.tags_for(clip["end"])["subject"] == "right"
    assert clip["lastFrame"] is not None and clip["lastFrame"].shape[2] == 3
