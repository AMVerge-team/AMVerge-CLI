from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from ...core.infra.ipc import log
from ...core.scenescout import embedding
from ...core.scenescout import indexing
from ...core.scenescout import search as scoutsearch
from ...core.scenescout.paths import db_path, list_db_paths
from ...core.scenescout.types import SearchOptions


def run_daemon(
    root: Optional[Path] = None,
    device: Optional[str] = None,
    idle_seconds: int = 300,
) -> None:
    resolved_device = embedding.resolve_device(device)
    log(f"[diag] scene scout daemon starting | device={resolved_device} idle_seconds={idle_seconds}")

    last_active = time.time()
    in_standby = False
    # a request is running; indexing outlasts idle_seconds, and moving the model to cpu
    # mid-request leaves its inputs on cuda and crashes the forward pass
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

    idle_thread = threading.Thread(target=_idle_checker, daemon=True)
    idle_thread.start()

    def _reply(data: dict) -> None:
        sys.stdout.write(json.dumps(data, separators=(",", ":")) + "\n")
        sys.stdout.flush()

    while True:
        # back here means the previous request finished, so idle time counts from now
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

        if action == "ping":
            _reply({"id": req_id, "status": "ok", "action": "pong"})
            continue

        if action == "standby":
            with lock:
                embedding.standby_model()
                in_standby = True
            _reply({"id": req_id, "status": "ok", "state": "standby"})
            continue

        if action == "activate":
            with lock:
                embedding.activate_model()
                in_standby = False
            _reply({"id": req_id, "status": "ok", "state": "active"})
            continue

        if action == "unload":
            with lock:
                embedding.unload_model()
                in_standby = False
            _reply({"id": req_id, "status": "ok", "state": "unloaded"})
            continue

        if action == "status":
            _reply(
                {
                    "id": req_id,
                    "status": "ok",
                    "loaded": embedding.is_model_loaded(),
                    "standby": in_standby,
                    "device": resolved_device,
                }
            )
            continue

        if action == "shutdown":
            _reply({"id": req_id, "status": "ok", "state": "shutdown"})
            stop_event.set()
            break

        if action == "search":
            query = req.get("query")
            image = req.get("image")
            raw_dbs = req.get("databases") or req.get("database") or []
            if isinstance(raw_dbs, str):
                raw_dbs = [raw_dbs]
            raw_videos = req.get("videos") or req.get("video") or []
            if isinstance(raw_videos, str):
                raw_videos = [raw_videos]

            top_k = int(req.get("top_k", 24))
            threshold = float(req.get("threshold", -1.0))
            include_thumbnails = bool(req.get("include_thumbnails", True))
            max_patches = int(req.get("max_patches", 256))
            search_device = req.get("device") or resolved_device

            targets = (
                [str(db_path(name, root)) for name in raw_dbs]
                if raw_dbs
                else [str(p) for p in list_db_paths(root)]
            )

            if not targets:
                _reply({"id": req_id, "status": "error", "error": "No databases found"})
                continue

            options = SearchOptions(
                top_k=top_k,
                similarity_threshold=threshold,
                include_thumbnails=include_thumbnails,
                max_patches=max_patches,
                device=search_device,
                databases=targets,
                video_paths=raw_videos,
            )

            if not embedding.is_model_loaded():
                _reply({"id": req_id, "status": "progress", "stage": "loading_model", "done": 0, "total": 1})

            try:
                started = time.perf_counter()
                hits = (
                    scoutsearch.search_image(str(image), options)
                    if image
                    else scoutsearch.search_text(query or "", options)
                )
                log(f"[diag] daemon search: {len(hits)} results in {time.perf_counter() - started:.3f}s")
                _reply({"id": req_id, "status": "ok", "results": [h.to_json() for h in hits]})
            except Exception as exc:
                _reply({"id": req_id, "status": "error", "error": str(exc)})
            continue

        if action == "add":
            video = req.get("video")
            database = req.get("database")
            detector = req.get("detector", "keyframe_detection")
            accurate = bool(req.get("accurate", False))
            max_patches = int(req.get("max_patches", 256))
            batch_size = int(req.get("batch_size", 16))
            no_thumbnails = bool(req.get("no_thumbnails", False))
            add_device = req.get("device") or resolved_device

            path = db_path(database, root)
            if not path.is_file():
                _reply({"id": req_id, "status": "error", "error": f"Database not found at {path}"})
                continue

            def on_progress(stage: str, done: int, total: int) -> None:
                _reply({"id": req_id, "status": "progress", "stage": stage, "done": done, "total": total})

            try:
                count = indexing.index_video(
                    path,
                    video,
                    device=add_device,
                    max_patches=max_patches,
                    batch_size=batch_size,
                    accurate=accurate,
                    method=detector,
                    generate_thumbnails=not no_thumbnails,
                    on_progress=on_progress,
                )
                _reply({"id": req_id, "status": "ok", "done": True, "scenes": count, "video": video})
            except Exception as exc:
                _reply({"id": req_id, "status": "error", "error": str(exc)})
            continue

        _reply({"id": req_id, "status": "error", "error": f"Unknown action: {action}"})

    stop_event.set()
    embedding.unload_model()
    log("[diag] scene scout daemon exited cleanly")
