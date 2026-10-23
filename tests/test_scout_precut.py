"""Scene ranges handed to `scout add --scenes-json` for an episode the app already cut."""

import json

from amverge.core.scenescout.indexing import load_precut_scenes, precut_scenes


def test_ranges_are_kept_as_millisecond_pairs():
    assert precut_scenes([[0, 4922], [5005, 9301]]) == [(0, 4922), (5005, 9301)]


def test_ranges_are_sorted_by_start():
    assert precut_scenes([[5005, 9301], [0, 4922]]) == [(0, 4922), (5005, 9301)]


def test_empty_and_malformed_ranges_are_dropped():
    raw = [[100, 100], [300, 200], [-5, 10], ["a", 9], [7], None, [0, 50]]
    assert precut_scenes(raw) == [(0, 50)]


def test_nothing_usable_is_an_empty_list():
    assert precut_scenes(None) == []
    assert precut_scenes([]) == []


def test_loads_from_a_json_file(tmp_path):
    path = tmp_path / "scenes.json"
    path.write_text(json.dumps([[0, 1001], [1001, 2500]]), encoding="utf-8")
    assert load_precut_scenes(path) == [(0, 1001), (1001, 2500)]
