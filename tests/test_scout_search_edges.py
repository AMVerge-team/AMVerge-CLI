import numpy as np

from amverge.core.scenescout import db as scoutdb
from amverge.core.scenescout.search import _search_one
from amverge.core.scenescout.types import SearchOptions

MINUTE = 60_000
SKIP = 2 * MINUTE


def vector(*values) -> bytes:
    return np.array(values, dtype=np.float32).tobytes()


def add_video(path, name, scenes):
    video_id = scoutdb.upsert_video(path, name, 1.0, status="indexing")
    rows = [(i, start, end, vector(1.0, 0.0), None) for i, (start, end) in enumerate(scenes)]
    scoutdb.insert_scenes(path, video_id, rows)
    scoutdb.mark_video_complete(path, video_id)


def starts(path, skip, top_k=50):
    options = SearchOptions(top_k=top_k, include_thumbnails=False, skip_edges_ms=skip)
    return sorted((h.video_path, h.start_ms) for h in _search_one(np.array([1.0, 0.0], dtype=np.float32), path, options))


def db(tmp_path):
    return scoutdb.create_database(str(tmp_path / "edges.scoutdb"))


def episode():
    return [(minute * MINUTE, (minute + 1) * MINUTE) for minute in range(24)]


def test_off_keeps_every_scene(tmp_path):
    path = db(tmp_path)
    add_video(path, "ep.mkv", episode())
    assert len(starts(path, 0)) == 24


def test_scenes_in_the_first_and_last_two_minutes_are_left_out(tmp_path):
    path = db(tmp_path)
    add_video(path, "ep.mkv", episode())
    assert [start for _, start in starts(path, SKIP)] == [m * MINUTE for m in range(2, 22)]


def test_a_scene_crossing_the_cutoff_is_left_out(tmp_path):
    path = db(tmp_path)
    add_video(path, "ep.mkv", [(0, SKIP + 1000), (SKIP + 1000, 10 * MINUTE), (10 * MINUTE, 24 * MINUTE - 1000), (24 * MINUTE - 1000, 24 * MINUTE)])
    assert [start for _, start in starts(path, SKIP)] == [SKIP + 1000]


def test_each_video_uses_its_own_length(tmp_path):
    path = db(tmp_path)
    add_video(path, "long.mkv", episode())
    add_video(path, "shorter.mkv", [(m * MINUTE, (m + 1) * MINUTE) for m in range(12)])
    kept = starts(path, SKIP)
    assert [s for v, s in kept if v == "shorter.mkv"] == [m * MINUTE for m in range(2, 10)]
    assert [s for v, s in kept if v == "long.mkv"] == [m * MINUTE for m in range(2, 22)]


def test_a_video_too_short_to_have_a_middle_keeps_everything(tmp_path):
    path = db(tmp_path)
    add_video(path, "clip.mp4", [(0, MINUTE), (MINUTE, 2 * MINUTE), (2 * MINUTE, 3 * MINUTE)])
    assert len(starts(path, SKIP)) == 3


def test_filtering_happens_before_the_result_limit(tmp_path):
    path = db(tmp_path)
    add_video(path, "ep.mkv", episode())
    assert len(starts(path, SKIP, top_k=20)) == 20
