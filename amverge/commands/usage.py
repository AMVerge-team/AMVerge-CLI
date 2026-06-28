from __future__ import annotations

from ..ui import banner, console, make_table


def usage() -> None:
    """Show command reference and usage examples."""
    banner("help")

    t = make_table(
        ("command",  "#22c55e bold", {"width": 10}),
        ("args",     "white",        {"width": 38}),
        ("note",     "bright_black", {}),
        title="Workflow commands",
    )
    t.add_row("detect",  "VIDEO  [--output DIR] [--method keyframe|edge]",  "split video into scenes")
    t.add_row("",        "[--min-duration 0.25] [--workers 4]",             "")
    t.add_row("",        "[--no-thumbnails] [--no-similarity]",             "")
    t.add_row("export",  "VIDEO  --scenes JSON  [--output DIR]",            "export scenes to disk")
    t.add_row("",        "[--select 0,2,5-8]  [--merge]  [--codec copy]",  "")
    t.add_row("merge",   "CLIP CLIP ...  --output FILE",                    "concat clips")
    t.add_row("info",    "VIDEO",                                            "show stream metadata")
    console.print(t)

    t2 = make_table(
        ("command",   "#22c55e bold", {"width": 10}),
        ("note",      "bright_black", {}),
        title="Info commands",
    )
    t2.add_row("usage",     "command reference (this page)")
    t2.add_row("about",     "what is AMVerge CLI")
    t2.add_row("credits",   "meet the team")
    t2.add_row("changelog", "version history")
    console.print(t2)

    console.print("\n[bright_black]  Examples[/]\n")
    t3 = make_table(
        ("",      "bright_black", {"width": 22}),
        ("",      "white",        {}),
    )
    t3.add_row("detect",             "amverge detect ep01.mkv")
    t3.add_row("detect (accurate)",  "amverge detect ep01.mkv --method edge --min-duration 0.5")
    t3.add_row("export all",         "amverge export ep01.mkv --scenes ep01_scenes/scenes.json")
    t3.add_row("export selection",   "amverge export ep01.mkv -s scenes.json --select 0,2,5-8 --merge")
    t3.add_row("merge",              "amverge merge scene_0001.mp4 scene_0002.mp4 -o out.mp4")
    t3.add_row("info",               "amverge info ep01.mkv")
    t3.add_row("library",            "from amverge import detect_scenes")
    console.print(t3)

    console.print("\n[bright_black]  Detection methods[/]\n")
    t4 = make_table(
        ("method",    "#22c55e bold", {"width": 10}),
        ("speed",     "white",        {"width": 8}),
        ("accuracy",  "white",        {"width": 10}),
        ("requires",  "bright_black", {}),
    )
    t4.add_row("keyframe", "fast",   "good",      "nothing extra")
    t4.add_row("edge",     "slower", "excellent", "pip install amverge-cli[edge]")
    console.print(t4)
    console.print()
