from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from ...core.infra.ipc import log
from ...core.scenescout import embedding
from ...core.scenescout import indexing
from ...core.scenescout import search as scoutsearch
from ...core.scenescout.paths import db_path, list_db_paths
from ...core.scenescout.types import SearchOptions


def _reply(data: dict) -> None:
    sys.stdout.write(json.dumps(data, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _as_list(value) -> list:
    if not value:
        return []
    return [value] if isinstance(value, str) else list(value)


def _search(req: dict, root: Optional[Path], device: str) -> dict:
    raw_dbs = _as_list(req.get("databases") or req.get("database"))
    targets = [str(db_path(name, root)) for name in raw_dbs] if raw_dbs else [str(p) for p in list_db_paths(root)]
    if not targets:
        return {"status": "error", "error": "No databases found"}

    options = SearchOptions(
        top_k=int(req.get("top_k", 24)),
        similarity_threshold=float(req.get("threshold", -1.0)),
        include_thumbnails=bool(req.get("include_thumbnails", True)),
        max_patches=int(req.get("max_patches", 256)),
        device=req.get("device") or device,
        databases=targets,
        video_paths=_as_list(req.get("videos") or req.get("video")),
        start_subjects=list(req.get("start_subjects") or []),
        include_unclear=bool(req.get("include_unclear", False)),
    )

    if not embedding.is_model_loaded():
        _reply({"id": req.get("id"), "status": "progress", "stage": "loading_model", "done": 0, "total": 1})

    query = req.get("query")
    image = req.get("image")
    flow = req.get("flow_from")
    started = time.perf_counter()
    reply: dict = {"status": "ok"}
    if flow:
        hits, reply["flow"] = scoutsearch.search_flow(
            str(flow["path"]),
            options,
            text=query or None,
            start_ms=flow.get("start_ms"),
            end_ms=flow.get("end_ms"),
        )
    elif image:
        hits = scoutsearch.search_image(str(image), options)
    else:
        hits = scoutsearch.search_text(query or "", options)
    log(f"[diag] daemon search: {len(hits)} results in {time.perf_counter() - started:.3f}s")
    reply["results"] = [h.to_json() for h in hits]
    return reply


def _add(req: dict, root: Optional[Path], device: str, on_progress: Callable[[str, int, int], None]) -> dict:
    video = req.get("video")
    path = db_path(req.get("database"), root)
    if not path.is_file():
        return {"status": "error", "error": f"Database not found at {path}"}

    count = indexing.index_video(
        path,
        video,
        device=req.get("device") or device,
        max_patches=int(req.get("max_patches", 256)),
        batch_size=int(req.get("batch_size", 16)),
        accurate=bool(req.get("accurate", False)),
        method=req.get("detector", "keyframe_detection"),
        generate_thumbnails=not bool(req.get("no_thumbnails", False)),
        motion=bool(req.get("motion", True)),
        scenes=indexing.precut_scenes(req.get("scenes")) or None,
        on_progress=on_progress,
    )
    return {"status": "ok", "done": True, "scenes": count, "video": video}


def run_daemon(
    root: Optional[Path] = None,
    device: Optional[str] = None,
    idle_seconds: int = 300,
) -> None:
    resolved_device = embedding.resolve_device(device)
    log(f"[diag] scene scout daemon starting | device={resolved_device} idle_seconds={idle_seconds}")

    last_active = time.time()
    in_standby = False
    busy = False
    lock = threading.Lock()
    stop_event = threading.Event()

    def _idle_checker() -> None:
        nonlocal in_standby
        while not stop_event.is_set():
            time.sleep(2.0)
            if idle_seconds <= 0:
                continue
            with lock:
                if not in_standby and not busy and embedding.is_model_loaded():
                    if time.time() - last_active >= idle_seconds:
                        embedding.standby_model()
                        in_standby = True

    threading.Thread(target=_idle_checker, daemon=True).start()

    while True:
        with lock:
            busy = False
            last_active = time.time()

        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except Exception as exc:
            _reply({"status": "error", "error": f"Invalid JSON: {exc}"})
            continue

        req_id = req.get("id")
        action = req.get("action", "")

        with lock:
            last_active = time.time()
            busy = True
            if in_standby:
                embedding.activate_model()
                in_standby = False

        if action == "shutdown":
            _reply({"id": req_id, "status": "ok", "state": "shutdown"})
            stop_event.set()
            break

        try:
            if action == "ping":
                reply = {"status": "ok", "action": "pong"}
            elif action == "standby":
                with lock:
                    embedding.standby_model()
                    in_standby = True
                reply = {"status": "ok", "state": "standby"}
            elif action == "activate":
                with lock:
                    embedding.activate_model()
                    in_standby = False
                reply = {"status": "ok", "state": "active"}
            elif action == "unload":
                with lock:
                    embedding.unload_model()
                    in_standby = False
                reply = {"status": "ok", "state": "unloaded"}
            elif action == "status":
                reply = {
                    "status": "ok",
                    "loaded": embedding.is_model_loaded(),
                    "standby": in_standby,
                    "device": resolved_device,
                }
            elif action == "search":
                reply = _search(req, root, resolved_device)
            elif action == "add":
                reply = _add(
                    req,
                    root,
                    resolved_device,
                    lambda stage, done, total: _reply(
                        {"id": req_id, "status": "progress", "stage": stage, "done": done, "total": total}
                    ),
                )
            else:
                reply = {"status": "error", "error": f"Unknown action: {action}"}
        except Exception as exc:
            reply = {"status": "error", "error": str(exc)}

        _reply({"id": req_id, **reply})

    stop_event.set()
    embedding.unload_model()
    log("[diag] scene scout daemon exited cleanly")
