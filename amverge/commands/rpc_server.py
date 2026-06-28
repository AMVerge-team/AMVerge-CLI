from __future__ import annotations

import json
import os
import sys
import threading
import time

import typer


def rpc_server() -> None:
    """Discord RPC server sidecar for the Tauri app.

    Reads JSON commands from stdin, updates Discord presence.
    Shuts down when stdin closes or a shutdown command is received.
    Called by Rust as a long-lived subprocess.
    """
    from ..core.discord_rpc import DiscordRPC, RPC_AVAILABLE

    if not RPC_AVAILABLE:
        raise typer.Exit(1)

    rpc = DiscordRPC()
    if not rpc.connect():
        raise typer.Exit(1)

    shutdown_event = threading.Event()
    last_update_time = 0.0
    last_details = ""

    def _monitor_stdin() -> None:
        nonlocal last_update_time, last_details
        while not shutdown_event.is_set():
            line = sys.stdin.readline()
            if not line:
                shutdown_event.set()
                break
            try:
                data = json.loads(line)
                cmd = data.get("type")
                if cmd == "update":
                    now = time.time()
                    details = data.get("details")
                    if now - last_update_time >= 15 or details != last_details:
                        rpc._update(
                            details=details,
                            state=data.get("state"),
                            large_image=data.get("large_image", "amverge_logo"),
                            large_text=data.get("large_text", "AMVerge"),
                            small_image=data.get("small_image"),
                            small_text=data.get("small_text"),
                            buttons=data.get("buttons", True),
                        )
                        last_update_time = now
                        last_details = details or ""
                elif cmd == "clear":
                    rpc.clear_presence()
                elif cmd in ("exit", "shutdown"):
                    shutdown_event.set()
                    break
            except Exception:
                pass

    thread = threading.Thread(target=_monitor_stdin, daemon=True)
    thread.start()

    parent_pid = os.getppid()

    try:
        while not shutdown_event.is_set():
            if os.name != "nt" and os.getppid() != parent_pid:
                shutdown_event.set()
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass

    rpc.clear_presence()
    rpc.disconnect()
