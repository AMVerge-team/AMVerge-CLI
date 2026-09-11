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
    from ...core.scenescout.paths import db_path

    path = db_path(database, root)
    if not path.is_file():
        scoutdb.create_database(database, root)

    def on_progress(stage: str, done: int, total: int) -> None:
        if as_json:
            print(json.dumps({"stage": stage, "done": done, "total": total}), flush=True)
        else:
            console.print(f"[dim]{stage}: {done}/{total}[/dim]")

    if not as_json:
        banner("scout add")

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
            on_progress=on_progress,
        )
    except NotImplementedError as exc:
        if as_json:
            print(json.dumps({"error": str(exc)}), flush=True)
            raise typer.Exit(1)
        fail(str(exc))
        raise typer.Exit(1)

    if as_json:
        print(json.dumps({"done": True, "scenes": count, "video": str(video)}), flush=True)
    else:
        console.print(f"Indexed [bold]{count}[/bold] scenes from {Path(video).name}")


@scout.command("search")
def search(
    query: Optional[str] = typer.Argument(None, help="What to look for"),
    database: list[str] = typer.Option([], "--db", help="Database to search (repeatable)"),
    image: Optional[Path] = typer.Option(None, "--image", help="Search by reference image instead"),
    root: Optional[Path] = typer.Option(None, "--root", help="Scene Scout storage folder"),
    top_k: int = typer.Option(24, "--top-k", help="Maximum results"),
    threshold: float = typer.Option(-1.0, "--threshold", help="Minimum score, -1 disables"),
    device: Optional[str] = typer.Option(None, "--device", help="cuda, mps or cpu"),
    max_patches: int = typer.Option(256, "--max-patches", help="Model patch budget"),
    no_thumbnails: bool = typer.Option(False, "--no-thumbnails", help="Omit thumbnails from output"),
    as_json: bool = typer.Option(False, "--json", help="Output as JSON"),
) -> None:
    """Search indexed scenes by description, or by reference image."""
    from ...core.scenescout import search as scoutsearch
    from ...core.scenescout.paths import db_path, list_db_paths
    from ...core.scenescout.types import SearchOptions

    if not query and not image:
        fail("Give a search query, or --image")
        raise typer.Exit(1)

    # no --db means every database in the root, which is what the app's
    # "search all" toggle sends
    targets = [str(db_path(name, root)) for name in database] if database else [
        str(p) for p in list_db_paths(root)
    ]
    if not targets:
        if as_json:
            _emit({"results": [], "error": "no databases"}, True)
            raise typer.Exit(1)
        fail("No databases to search. Create one with: amverge scout create <name>")
        raise typer.Exit(1)

    options = SearchOptions(
        top_k=top_k,
        similarity_threshold=threshold,
        include_thumbnails=not no_thumbnails,
        max_patches=max_patches,
        device=device,
        databases=targets,
    )

    try:
        hits = (
            scoutsearch.search_image(str(image), options)
            if image
            else scoutsearch.search_text(query or "", options)
        )
    except NotImplementedError as exc:
        if as_json:
            _emit({"results": [], "error": str(exc)}, True)
            raise typer.Exit(1)
        fail(str(exc))
        raise typer.Exit(1)

    if as_json:
        _emit({"results": [h.to_json() for h in hits]}, True)
        return

    banner("scout search")
    if not hits:
        console.print("[dim]No matches.[/dim]")
        return

    table = make_table(
        ("Score", "#22c55e", {"justify": "right"}),
        ("Video", "bold", {}),
        ("Scene", None, {"justify": "right"}),
        ("Start", None, {"justify": "right"}),
        ("Database", "bright_black", {}),
    )
    for hit in hits:
        table.add_row(
            f"{hit.score:.3f}",
            Path(hit.video_path).name,
            str(hit.scene_index),
            f"{hit.start_ms / 1000:.1f}s",
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

    if as_json:
        _emit(payload, True)
        return

    banner("scout status")
    console.print(f"Model installed : {'yes' if available else 'no'}")
    console.print(f"Model version   : {payload['modelVersion']}")
    console.print(f"Device          : {device or 'n/a'}")
    console.print(f"Storage root    : {directory}")
    console.print(f"Databases       : {payload['databaseCount']}")
    if not available:
        console.print(r"[yellow]Install the model extra:[/yellow] pip install 'amverge\[scout]'")
