from __future__ import annotations

from ..ui import banner, console, make_table


def usage() -> None:
    """Show command reference and usage examples."""
    banner("help")

    t = make_table(
        ("command",  "#22c55e bold", {"width": 10}),
        ("args",     "white",        {"width": 42}),
        ("note",     "bright_black", {}),
        title="Workflow commands",
    )
    t.add_row("detect",  "VIDEO  [--output DIR] [--method keyframe|edge]",  "split video into scenes")
    t.add_row("",        "[--min-duration 0.25] [--workers 4]",             "")
    t.add_row("",        "[--no-thumbnails] [--no-similarity] [--ipc]",     "")
    t.add_row("export",  "VIDEO  --scenes JSON  [--output DIR]",            "export scenes to disk")
    t.add_row("",        "[--select 0,2,5-8]  [--merge]  [--codec copy]",  "")
    t.add_row("merge",   "CLIP CLIP ...  --output FILE",                    "concat clips")
    console.print(t)

    t2 = make_table(
        ("command",    "#22c55e bold", {"width": 12}),
        ("args",       "white",        {"width": 42}),
        ("note",       "bright_black", {}),
        title="Video info commands",
    )
    t2.add_row("info",       "VIDEO",                                         "stream metadata (PyAV)")
    t2.add_row("probe",      "VIDEO  [--no-keyframes] [--cache-dir DIR]",     "V2 diagnostics: codec, HEVC, keyframes, cache")
    t2.add_row("gpu",        "",                                               "PyTorch, CUDA, GPU, and optional deps status")
    t2.add_row("keyframes",  "VIDEO  [--json] [--count]",                     "dump keyframe timestamps")
    t2.add_row("scenes",     "VIDEO  [--json] [--min-duration N]",            "scene list from .npy cache")
    t2.add_row("",           "[--cache-dir DIR]",                             "")
    t2.add_row("cache",      "DIR  [--clear VIDEO] [--clear-all]",            "list or delete .npy scene caches")
    console.print(t2)

    t3 = make_table(
        ("command",    "#22c55e bold", {"width": 12}),
        ("note",       "bright_black", {}),
        title="App commands",
    )
    t3.add_row("usage",     "command reference (this page)")
    t3.add_row("about",     "what is AMVerge CLI")
    t3.add_row("credits",   "meet the team")
    t3.add_row("changelog", "version history")
    console.print(t3)

    console.print("\n[bright_black]  Examples[/]\n")
    t4 = make_table(
        ("",      "bright_black", {"width": 24}),
        ("",      "white",        {}),
    )
    t4.add_row("detect (V2 ML)",        "amverge detect ep01.mkv")
    t4.add_row("detect (keyframe)",     "amverge detect ep01.mkv --method keyframe")
    t4.add_row("detect (edge)",         "amverge detect ep01.mkv --method edge --min-duration 0.5")
    t4.add_row("export all",            "amverge export ep01.mkv --scenes scenes.json")
    t4.add_row("export selection",      "amverge export ep01.mkv -s scenes.json --select 0,2,5-8 --merge")
    t4.add_row("merge",                 "amverge merge scene_0001.mp4 scene_0002.mp4 -o out.mp4")
    t4.add_row("probe",                 "amverge probe ep01.mkv")
    t4.add_row("gpu check",             "amverge gpu")
    t4.add_row("probe (skip keyframes)","amverge probe ep01.mkv --no-keyframes")
    t4.add_row("keyframes",             "amverge keyframes ep01.mkv --json | python parse.py")
    t4.add_row("scenes from cache",     "amverge scenes ep01.mkv --cache-dir %APPDATA%/amverge/episodes/id/")
    t4.add_row("list caches",           "amverge cache %APPDATA%/amverge/episodes/id/")
    t4.add_row("clear cache",           "amverge cache ./out --clear ep01.mkv")
    t4.add_row("library",               "from amverge import detect_scenes")
    console.print(t4)

    console.print("\n[bright_black]  Detection methods[/]\n")
    t5 = make_table(
        ("method",      "#22c55e bold", {"width": 12}),
        ("speed",       "white",        {"width": 8}),
        ("accuracy",    "white",        {"width": 10}),
        ("requires",    "bright_black", {}),
    )
    t5.add_row("transnetv2", "medium", "best",      "pip install amverge[ml]  (CUDA optional)")
    t5.add_row("keyframe",   "fast",   "good",      "nothing extra")
    t5.add_row("edge",       "slower", "excellent", "pip install amverge[edge]")
    console.print(t5)
    console.print()
