from __future__ import annotations

from ...ui import banner, console, make_table


def usage() -> None:
    """Show command reference and usage examples."""
    banner("help")

    t = make_table(
        ("command",  "#22c55e bold", {"width": 10}),
        ("args",     "white",        {"width": 42}),
        ("note",     "bright_black", {}),
        title="Workflow commands",
    )
    t.add_row("detect",  "VIDEO  [--output DIR] [--method keyframe|edge|transnetv2]",  "split video into scenes")
    t.add_row("",        "[--decode-method ffmpeg|nelux] [--min-duration 0.25]",  "")
    t.add_row("",        "[--workers 4] [--no-thumbnails] [--no-similarity] [--ipc]",  "")
    t.add_row("export",  "VIDEO  --scenes JSON  [--output DIR]",               "export scenes to disk")
    t.add_row("",        "[--select 0,2,5-8]  [--merge]  [--codec copy]",     "")
    t.add_row("",        "[--audio copy] [--container mp4] [--hardware auto]", "")
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
    t2.add_row("doctor",     "",                                               "full environment health check")
    t2.add_row("version",    "[--json]",                                       "CLI, Python, and dep versions")
    t2.add_row("bench",      "VIDEO  [--skip-ml]",                             "benchmark keyframe scan and TransNetV2 inference")
    t2.add_row("keyframes",  "VIDEO  [--json] [--count]",                     "dump keyframe timestamps")
    t2.add_row("scenes",     "VIDEO  [--json] [--min-duration N]",            "scene list from .npy cache")
    t2.add_row("",           "[--cache-dir DIR]",                             "")
    t2.add_row("cache",      "DIR  [--clear VIDEO] [--clear-all]",            "list or delete .npy scene caches")
    console.print(t2)

    t3 = make_table(
        ("command",    "#22c55e bold", {"width": 13}),
        ("args",       "white",        {"width": 42}),
        ("note",       "bright_black", {}),
        title="Processing commands",
    )
    t3.add_row("upscale",    "VIDEO  [--model KEY] [--scale 2|4] [--preset high]", "AI super-resolution upscaling")
    t3.add_row("",           "[--mode light|medium|strong] [--fit-w W] [--fit-h H]", "")
    t3.add_row("interpolate","VIDEO  [--model KEY] [--factor 2-64] [--preset high]", "RIFE AI frame interpolation")
    t3.add_row("",           "[--target-size-mb N] [--no-monitor]", "")
    t3.add_row("flowframes", "VIDEO  [--factor 2-64] [--ai engine] [--model name]", "Flowframes 1.42.0 interpolation")
    t3.add_row("models",     "[--upscale] [--interpolation] [--download KEY]", "manage upscale + interpolation model files")
    t3.add_row("",           "[--delete KEY] [--storage] [--verbose]", "")
    console.print(t3)

    t4 = make_table(
        ("command",    "#22c55e bold", {"width": 12}),
        ("note",       "bright_black", {}),
        title="App commands",
    )
    t4.add_row("usage",     "command reference (this page)")
    t4.add_row("about",     "what is AMVerge CLI")
    t4.add_row("credits",   "meet the team")
    t4.add_row("changelog", "version history")
    console.print(t4)

    console.print("\n[bright_black]  Examples[/]\n")
    t5 = make_table(
        ("",      "bright_black", {"width": 24}),
        ("",      "white",        {}),
    )
    t5.add_row("detect (V2 ML)",        "amverge detect ep01.mkv")
    t5.add_row("detect (keyframe)",     "amverge detect ep01.mkv --method keyframe")
    t5.add_row("detect (edge)",         "amverge detect ep01.mkv --method edge --min-duration 0.5")
    t5.add_row("detect (nelux GPU)",    "amverge detect ep01.mkv --method transnetv2 --decode-method nelux")
    t5.add_row("export all",            "amverge export ep01.mkv --scenes scenes.json")
    t5.add_row("export selection",      "amverge export ep01.mkv -s scenes.json --select 0,2,5-8 --merge")
    t5.add_row("merge",                 "amverge merge scene_0001.mp4 scene_0002.mp4 -o out.mp4")
    t5.add_row("probe",                 "amverge probe ep01.mkv")
    t5.add_row("gpu check",             "amverge gpu")
    t5.add_row("probe (skip keyframes)","amverge probe ep01.mkv --no-keyframes")
    t5.add_row("keyframes",             "amverge keyframes ep01.mkv --json | python parse.py")
    t5.add_row("scenes from cache",     "amverge scenes ep01.mkv --cache-dir %APPDATA%/amverge/episodes/id/")
    t5.add_row("list caches",           "amverge cache %APPDATA%/amverge/episodes/id/")
    t5.add_row("clear cache",           "amverge cache ./out --clear ep01.mkv")
    t5.add_row("library",               "from amverge import detect_scenes")
    t5.add_row("upscale (ML)",          "amverge upscale ep01.mkv -m adore -s 2")
    t5.add_row("upscale (Anime4K)",     "amverge upscale ep01.mkv -m anime4k --mode medium -s 2")
    t5.add_row("interpolate",           "amverge interpolate ep01.mkv -m rife4.25 -f 2")
    t5.add_row("interpolate (archival)","amverge interpolate ep01.mkv -m rife4.25-heavy -f 4 -p archival")
    t5.add_row("list models",           "amverge models")
    t5.add_row("list interpolation","amverge models --interpolation")
    t5.add_row("download model",        "amverge models --download rife4.25-heavy")
    console.print(t5)

    console.print("\n[bright_black]  Detection methods[/]\n")
    t6 = make_table(
        ("method",      "#22c55e bold", {"width": 12}),
        ("speed",       "white",        {"width": 8}),
        ("accuracy",    "white",        {"width": 10}),
        ("requires",    "bright_black", {}),
    )
    t6.add_row("transnetv2", "medium", "best",      "pip install amverge[ml]  (CUDA optional)")
    t6.add_row("keyframe",   "fast",   "good",      "nothing extra")
    t6.add_row("edge",       "slower", "excellent", "pip install amverge[edge]")
    console.print(t6)
    console.print()
