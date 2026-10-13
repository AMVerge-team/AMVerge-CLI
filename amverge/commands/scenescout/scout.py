"""`amverge scout` - natural-language scene search.

A Typer sub-app rather than a flat command, because Scene Scout has several
verbs that share one storage root. It is registered on the main app in
``amverge/cli.py``.

Every subcommand takes ``--root`` and ``--json``. Those two flags are the whole
app integration: AMVerge passes the storage folder the user chose in Settings
and parses the JSON, and a developer running the CLI by hand omits both and gets
a table against the standalone default.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import typer

from ...ui import banner, console, make_table, fail

scout = typer.Typer(
    name="scout",
    help="Search video scenes by description (Scene Scout).",
    no_args_is_help=True,
)


def _emit(payload: object, as_json: bool) -> None:
    """JSON goes to stdout verbatim, so the app can parse it without stripping."""
    if as_json:
        print(json.dumps(payload, indent=None, separators=(",", ":")))


@scout.command("databases")
def databases(
    root: Optional[Path] = typer.Option(None, "--root", help="Scene Scout storage folder"),
    as_json: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """List every Scene Scout database and what it holds."""
    from ...core.scenescout import db as scoutdb

    infos = scoutdb.list_databases(root)

    if as_json:
        _emit({"databases": [i.to_json() for i in infos]}, True)
        return

    banner("scout databases")
    if not infos:
        console.print("[dim]No databases yet. Create one with: amverge scout create <name>[/dim]")
        return

    table = make_table(
        ("Name", "bold", {}),
        ("Videos", None, {"justify": "right"}),
        ("Scenes", None, {"justify": "right"}),
        ("Size", None, {"justify": "right"}),
        ("Path", "bright_black", {"overflow": "fold"}),
    )
    for info in infos:
        table.add_row(
            info.name,
            str(info.video_count),
            str(info.scene_count),
            f"{info.size_bytes / 1_048_576:.1f} MB",
            info.path,
        )
    console.print(table)


@scout.command("create")
def create(
    name: str = typer.Argument(..., help="Database name"),
    root: Optional[Path] = typer.Option(None, "--root", help="Scene Scout storage folder"),
    as_json: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Create an empty database."""
    from ...core.scenescout import db as scoutdb

    path = scoutdb.create_database(name, root)
    info = scoutdb.database_info(path)

    if as_json:
        _emit({"database": info.to_json()}, True)
        return

    banner("scout create")
    console.print(f"Created [bold]{info.name}[/bold]")
    console.print(f"[dim]{info.path}[/dim]")


@scout.command("delete")
def delete(
    name: str = typer.Argument(..., help="Database name"),
    root: Optional[Path] = typer.Option(None, "--root", help="Scene Scout storage folder"),
    as_json: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Delete a database and everything indexed into it."""
    from ...core.scenescout import db as scoutdb

    removed = scoutdb.delete_database(name, root)

    if as_json:
        _emit({"deleted": removed, "name": name}, True)
        return

    banner("scout delete")
    if removed:
        console.print(f"Deleted [bold]{name}[/bold]")
    else:
        fail(f"No database named {name}")
        raise typer.Exit(1)


@scout.command("info")
def info(
    database: str = typer.Argument(..., help="Database name or path"),
    root: Optional[Path] = typer.Option(None, "--root", help="Scene Scout storage folder"),
    as_json: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Stats for one database, including one saved outside the storage folder."""
    from ...core.scenescout import db as scoutdb
    from ...core.scenescout.paths import db_path

    path = db_path(database, root)
    if not path.is_file():
        if as_json:
            _emit({"error": "database not found", "path": str(path)}, True)
            raise typer.Exit(1)
        fail(f"No database at {path}")
        raise typer.Exit(1)

    result = scoutdb.database_info(path)
    if as_json:
        _emit({"database": result.to_json()}, True)
        return

    banner("scout info")
    console.print(f"Name   : {result.name}")
    console.print(f"Path   : {result.path}")
    console.print(f"Videos : {result.video_count}")
    console.print(f"Scenes : {result.scene_count}")


@scout.command("open")
def open_db(
    path: str = typer.Argument(..., help="Path to database file"),
    root: Optional[Path] = typer.Option(None, "--root", help="Scene Scout storage folder"),
    as_json: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    from ...core.scenescout import db as scoutdb
    from ...core.scenescout.paths import db_path

    resolved = db_path(path, root)
    try:
        result = scoutdb.open_database(resolved)
    except Exception as e:
        if as_json:
            _emit({"error": str(e), "path": str(resolved)}, True)
            raise typer.Exit(1)
        fail(str(e))
        raise typer.Exit(1)

    if as_json:
        _emit({"database": result.to_json()}, True)
        return

    banner("scout open")
    console.print(f"Name   : {result.name}")
    console.print(f"Path   : {result.path}")
    console.print(f"Videos : {result.video_count}")
    console.print(f"Scenes : {result.scene_count}")


@scout.command("videos")
def videos(
    database: str = typer.Argument(..., help="Database name"),
    root: Optional[Path] = typer.Option(None, "--root", help="Scene Scout storage folder"),
    as_json: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """List the videos indexed into a database."""
    from ...core.scenescout import db as scoutdb
    from ...core.scenescout.paths import db_path

    path = db_path(database, root)
    if not path.is_file():
        if as_json:
            _emit({"videos": [], "error": "database not found"}, True)
            raise typer.Exit(1)
        fail(f"No database named {database}")
        raise typer.Exit(1)

    entries = scoutdb.list_videos(path)

    if as_json:
        _emit({"videos": [v.to_json() for v in entries]}, True)
        return

    banner("scout videos")
    if not entries:
        console.print("[dim]Nothing indexed yet.[/dim]")
        return

    table = make_table(
        ("ID", None, {"justify": "right"}),
        ("Name", "bold", {}),
        ("Scenes", None, {"justify": "right"}),
        ("Status", None, {}),
    )
    for video in entries:
        table.add_row(str(video.id), Path(video.filepath).name, str(video.scene_count), video.status)
    console.print(table)


@scout.command("delete-video")
def delete_video(
    database: str = typer.Argument(..., help="Database name or path"),
    video_identifier: Optional[str] = typer.Argument(None, help="Video ID or filepath to remove"),
    id: Optional[int] = typer.Option(None, "--id", help="Video ID to remove"),
    video: Optional[str] = typer.Option(None, "--video", help="Video filepath to remove"),
    root: Optional[Path] = typer.Option(None, "--root", help="Scene Scout storage folder"),
    as_json: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    from ...core.scenescout import db as scoutdb
    from ...core.scenescout.paths import db_path

    if id is None and not video and video_identifier is not None:
        if video_identifier.isdigit():
            id = int(video_identifier)
        else:
            video = video_identifier

    path = db_path(database, root)
    if not path.is_file():
        if as_json:
            _emit({"deleted": False, "error": f"database not found at {path}"}, True)
            raise typer.Exit(1)
        fail(f"No database at {path}")
        raise typer.Exit(1)

    if id is None and not video:
        if as_json:
            _emit({"deleted": False, "error": "specify --id or --video to delete"}, True)
            raise typer.Exit(1)
        fail("Specify a video ID or filepath to delete")
        raise typer.Exit(1)

    removed = scoutdb.remove_video(path, video_id=id, filepath=video)

    if as_json:
        _emit({"deleted": removed, "database": str(path), "id": id, "video": video}, True)
        return

    banner("scout delete-video")
    if removed:
        console.print(f"Removed video from [bold]{path.name}[/bold]")
    else:
        fail(f"Video not found in {path.name}")
        raise typer.Exit(1)


scout.command("remove-video", hidden=True)(delete_video)


@scout.command("add")
def add(
    video: Path = typer.Argument(..., help="Video file to index", exists=True),
    database: str = typer.Option(..., "--db", help="Database name"),
    root: Optional[Path] = typer.Option(None, "--root", help="Scene Scout storage folder"),
    device: Optional[str] = typer.Option(None, "--device", help="cuda, mps or cpu"),
    max_patches: int = typer.Option(256, "--max-patches", help="Model patch budget"),
    batch_size: int = typer.Option(16, "--batch-size", help="Frames per inference batch"),
    accurate: bool = typer.Option(False, "--accurate", help="Slower, more precise scene detection"),
    detector: str = typer.Option(
        "keyframe_detection",
        "--detector",
        help="Scene detector: keyframe_detection or transnetv2_gpu",
    ),
    no_thumbnails: bool = typer.Option(False, "--no-thumbnails", help="Skip thumbnail generation"),
    no_motion: bool = typer.Option(False, "--no-motion", help="Skip start/end motion tags"),
    as_json: bool = typer.Option(False, "--json", help="Emit progress and result as JSON lines"),
) -> None:
    """Index a video into a database.

    With ``--json`` this emits one JSON object per line: ``{"stage":...}``
    progress lines followed by a final ``{"done":true,...}``. The app reads it
    line by line to drive its progress bar, which is why progress is
    newline-delimited rather than one document at the end.
    """
    from ...core.scenescout import db as scoutdb
    from ...core.scenescout import indexing
    from ...core.scenescout.embedding import SceneScoutModelUnavailable
    from ...core.scenescout.paths import db_path
    from ...core.infra.ipc import log

    path = db_path(database, root)
    if not path.is_file() and root is None:
        scoutdb.create_database(database, root)

    if not path.is_file():
        if as_json:
            print(
                json.dumps(
                    {
                        "error": f"database not found at {path}; create it first with: amverge scout create <name>"
                    }
                ),
                flush=True,
            )
            raise typer.Exit(1)
        fail(f"No database at {path}. Create it first with: amverge scout create <name>")
        raise typer.Exit(1)

    def on_progress(stage: str, done: int, total: int) -> None:
        if as_json:
            print(json.dumps({"stage": stage, "done": done, "total": total}), flush=True)
        else:
            console.print(f"[dim]{stage}: {done}/{total}[/dim]")

    if not as_json:
        banner("scout add")

    started = time.perf_counter()
    log(
        f"[diag] scene scout add | db={Path(database).name} video={Path(video).name} "
        f"detector={detector} accurate={accurate}"
    )
    try:
        count = indexing.index_video(
            path,
            video,
            device=device,
            max_patches=max_patches,
            batch_size=batch_size,
            accurate=accurate,
            method=detector,
            generate_thumbnails=not no_thumbnails,
            motion=not no_motion,
            on_progress=on_progress,
        )
    except (NotImplementedError, SceneScoutModelUnavailable) as exc:
        if as_json:
            print(json.dumps({"error": str(exc)}), flush=True)
            raise typer.Exit(1)
        fail(str(exc))
        raise typer.Exit(1)
    log(f"[diag] scene scout add done: {count} scenes in {time.perf_counter() - started:.2f}s")

    if as_json:
        print(json.dumps({"done": True, "scenes": count, "video": str(video)}), flush=True)
    else:
        console.print(f"Indexed [bold]{count}[/bold] scenes from {Path(video).name}")


@scout.command("search")
def search(
    query: Optional[str] = typer.Argument(None, help="What to look for"),
    database: list[str] = typer.Option([], "--db", help="Database to search (repeatable)"),
    video: list[str] = typer.Option([], "--video", help="Filter search to specific video filepath(s)"),
    image: Optional[Path] = typer.Option(None, "--image", help="Search by reference image instead"),
    root: Optional[Path] = typer.Option(None, "--root", help="Scene Scout storage folder"),
    top_k: int = typer.Option(24, "--top-k", help="Maximum results"),
    threshold: float = typer.Option(-1.0, "--threshold", help="Minimum score, -1 disables"),
    device: Optional[str] = typer.Option(None, "--device", help="cuda, mps or cpu"),
    max_patches: int = typer.Option(256, "--max-patches", help="Model patch budget"),
    no_thumbnails: bool = typer.Option(False, "--no-thumbnails", help="Omit thumbnails from output"),
    flow_from: Optional[Path] = typer.Option(
        None, "--flow-from", help="Find scenes that flow out of this clip: most similar to its last frame", exists=True
    ),
    flow_start_ms: Optional[int] = typer.Option(None, "--flow-start-ms", help="Range start inside --flow-from"),
    flow_end_ms: Optional[int] = typer.Option(None, "--flow-end-ms", help="Range end inside --flow-from"),
    start_subject: list[str] = typer.Option(
        [], "--start-subject", help="Keep scenes whose subject starts moving this way (repeatable)"
    ),
    include_unclear: bool = typer.Option(False, "--include-unclear", help="Keep scenes with unreadable motion"),
    as_json: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Search indexed scenes by description, reference image, or a clip to continue."""
    from ...core.scenescout import search as scoutsearch
    from ...core.scenescout.embedding import SceneScoutModelUnavailable
    from ...core.scenescout.motion import SUBJECT_TAGS
    from ...core.scenescout.paths import db_path, list_db_paths
    from ...core.scenescout.types import SearchOptions
    from ...core.infra.ipc import log

    if not query and not image and not flow_from:
        fail("Give a search query, --image, or --flow-from")
        raise typer.Exit(1)
    unknown = [s for s in start_subject if s not in SUBJECT_TAGS]
    if unknown:
        fail(f"Unknown --start-subject {unknown}; use one of {', '.join(SUBJECT_TAGS)}")
        raise typer.Exit(1)

    targets = [str(db_path(name, root)) for name in database] if database else [
        str(p) for p in list_db_paths(root)
    ]
    if not targets:
        if as_json:
            _emit({"results": [], "error": "no databases"}, True)
            raise typer.Exit(1)
        fail("No databases to search. Create one with: amverge scout create <name>")
        raise typer.Exit(1)

    started = time.perf_counter()
    focus = query if query else (f"image={Path(image).name}" if image else "")
    if flow_from:
        focus = f"{focus} flow-from={Path(flow_from).name}".strip()
    log(
        f"[diag] scene scout search | {focus} databases={len(targets)} "
        f"top_k={top_k} threshold={threshold}"
    )

    options = SearchOptions(
        top_k=top_k,
        similarity_threshold=threshold,
        include_thumbnails=not no_thumbnails,
        max_patches=max_patches,
        device=device,
        databases=targets,
        video_paths=video,
        start_subjects=start_subject,
        include_unclear=include_unclear,
    )

    flow_tags = None
    try:
        if flow_from:
            hits, flow_tags = scoutsearch.search_flow(
                str(flow_from), options, text=query, start_ms=flow_start_ms, end_ms=flow_end_ms
            )
        elif image:
            hits = scoutsearch.search_image(str(image), options)
        else:
            hits = scoutsearch.search_text(query or "", options)
    except (NotImplementedError, SceneScoutModelUnavailable, ValueError) as exc:
        if as_json:
            _emit({"results": [], "error": str(exc)}, True)
            raise typer.Exit(1)
        fail(str(exc))
        raise typer.Exit(1)
    log(f"[diag] scene scout search done: {len(hits)} results in {time.perf_counter() - started:.2f}s")

    if as_json:
        payload: dict = {"results": [h.to_json() for h in hits]}
        if flow_tags is not None:
            payload["flow"] = flow_tags
        _emit(payload, True)
        return

    if flow_tags is not None:
        end = flow_tags["end"]
        console.print(f"Clip ends: subject {end['subject']} {end['speed'] or ''} / camera {end['camera']}")

    banner("scout search")
    if not hits:
        console.print("[dim]No matches.[/dim]")
        return

    table = make_table(
        ("Score", "#22c55e", {"justify": "right"}),
        ("Video", "bold", {}),
        ("Scene", None, {"justify": "right"}),
        ("Start", None, {"justify": "right"}),
        ("Starts moving", None, {}),
        ("Database", "bright_black", {}),
    )
    for hit in hits:
        start = hit.motion["start"] if hit.motion else None
        table.add_row(
            f"{hit.score:.3f}",
            Path(hit.video_path).name,
            str(hit.scene_index),
            f"{hit.start_ms / 1000:.1f}s",
            f"{start['subject']} / {start['camera']}" if start else "-",
            hit.database,
        )
    console.print(table)


@scout.command("status")
def status(
    root: Optional[Path] = typer.Option(None, "--root", help="Scene Scout storage folder"),
    as_json: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Report whether the model is installed and where databases live.

    The app calls this before showing the Scene Scout page, so it can explain a
    missing model instead of failing on the first search.
    """
    from ...core.scenescout import embedding
    from ...core.scenescout.paths import resolve_root, list_db_paths
    from ...core.infra.ipc import log

    available = embedding.is_available()
    directory = resolve_root(root)

    device = None
    if available:
        try:
            device = embedding.resolve_device()
        except Exception:
            device = None

    payload = {
        "modelAvailable": available,
        "modelVersion": embedding.model_version(),
        "device": device,
        "root": str(directory),
        "databaseCount": len(list_db_paths(root)),
    }

    log(
        f"[diag] scene scout status | model={'available' if available else 'not installed'} "
        f"device={device or 'n/a'} databases={payload['databaseCount']}"
    )

    if as_json:
        _emit(payload, True)
        return

    banner("scout status")
    console.print(f"Model installed : {'yes' if available else 'no'}")
    console.print(f"Model version   : {payload['modelVersion']}")
    console.print(f"Device          : {device or 'n/a'}")
    console.print(f"Storage root    : {directory}")
    if not available:
        console.print(r"[yellow]Install the model extra:[/yellow] pip install 'amverge\[scout]'")


@scout.command("daemon")
def daemon(
    root: Optional[Path] = typer.Option(None, "--root", help="Scene Scout storage folder"),
    device: Optional[str] = typer.Option(None, "--device", help="cuda, mps or cpu"),
    idle_seconds: int = typer.Option(300, "--idle-seconds", help="Idle seconds before model standby"),
) -> None:
    from .daemon import run_daemon

    run_daemon(root=root, device=device, idle_seconds=idle_seconds)


@scout.command("motion")
def motion_cmd(
    database: str = typer.Option(..., "--db", help="Database name or path"),
    root: Optional[Path] = typer.Option(None, "--root", help="Scene Scout storage folder"),
    as_json: bool = typer.Option(False, "--json", help="Emit progress and result as JSON lines"),
) -> None:
    """Measure start/end motion for indexed scenes that lack it (no model needed)."""
    from ...core.scenescout import indexing
    from ...core.scenescout.paths import db_path
    from ...core.infra.ipc import log

    path = db_path(database, root)
    if not path.is_file():
        if as_json:
            _emit({"error": f"database not found at {path}"}, True)
            raise typer.Exit(1)
        fail(f"Database does not exist: {path}")
        raise typer.Exit(1)

    def on_progress(stage: str, done: int, total: int) -> None:
        if as_json:
            print(json.dumps({"stage": stage, "done": done, "total": total}, separators=(",", ":")), flush=True)

    started = time.perf_counter()
    try:
        count = indexing.backfill_motion(path, on_progress=on_progress)
    except ImportError as exc:
        if as_json:
            _emit({"error": f"Motion tags need OpenCV: {exc}"}, True)
            raise typer.Exit(1)
        fail(f"Motion tags need OpenCV: {exc}")
        raise typer.Exit(1)
    log(f"[diag] scene scout motion | {path.name}: {count} scenes in {time.perf_counter() - started:.1f}s")

    if as_json:
        _emit({"done": True, "scenes": count, "database": str(path)}, True)
        return
    banner("scout motion")
    console.print(f"Measured motion for [bold]{count}[/bold] scenes in {path.name}")


@scout.command("clip-motion")
def clip_motion(
    clip: Path = typer.Argument(..., help="Clip, or a video with --start-ms/--end-ms", exists=True),
    start_ms: Optional[int] = typer.Option(None, "--start-ms", help="Range start"),
    end_ms: Optional[int] = typer.Option(None, "--end-ms", help="Range end"),
    as_json: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Start/end motion tags of one clip, for the app's flow chip (no model needed)."""
    from ...core.scenescout import motion

    try:
        measured = motion.measure_clip(str(clip), start_ms, end_ms)
    except (ValueError, ImportError) as exc:
        if as_json:
            _emit({"error": str(exc)}, True)
            raise typer.Exit(1)
        fail(str(exc))
        raise typer.Exit(1)
    tags = motion.scene_tags(measured["start"], measured["end"])

    if as_json:
        _emit({"motion": tags}, True)
        return
    banner("scout clip-motion")
    for side in ("start", "end"):
        t = tags[side]
        console.print(f"{side:<5}: subject {t['subject']} {t['speed'] or ''} / camera {t['camera']}")


@scout.command("generate-thumbnails")
def generate_thumbnails_cmd(
    database: str = typer.Option(..., "--db", help="Database name or path"),
    root: Optional[Path] = typer.Option(None, "--root", help="Scene Scout storage folder"),
    as_json: bool = typer.Option(False, "--json", help="Emit progress and result as JSON lines"),
) -> None:
    from ...core.scenescout import indexing
    from ...core.scenescout.paths import db_path
    from ...core.infra.ipc import log

    path = db_path(database, root)
    if not path.is_file():
        fail(f"Database does not exist: {path}")
        raise typer.Exit(1)

    log(f"[diag] scene scout generate-thumbnails | starting for {path.name}...")

    def on_progress(stage: str, done: int, total: int) -> None:
        if as_json:
            print(json.dumps({"stage": stage, "done": done, "total": total}, separators=(",", ":")), flush=True)

    count = indexing.generate_missing_thumbnails(path, on_progress=on_progress)
    log(f"[diag] scene scout generate-thumbnails | done: generated {count} thumbnails")

    if as_json:
        _emit({"done": True, "generated": count, "database": str(path)}, True)
        return

    banner("scout generate-thumbnails")
    console.print(f"[green]> Generated {count} missing or low-resolution thumbnails for {path.name}[/green]")

