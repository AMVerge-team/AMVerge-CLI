from __future__ import annotations

import threading
from typing import Optional, Any

try:
    from pypresence.presence import Presence
    from pypresence import exceptions as rpc_exceptions
    RPC_AVAILABLE = True
except ImportError:
    Presence = None
    rpc_exceptions = None
    RPC_AVAILABLE = False

CLIENT_ID = "1497922104065134823"


class DiscordRPC:
    def __init__(self, client_id: str = CLIENT_ID) -> None:
        self.client_id = client_id
        self.rpc: Optional[Any] = None
        self.connected = False
        self._lock = threading.Lock()

    def connect(self) -> bool:
        if not RPC_AVAILABLE or Presence is None:
            return False
        if self.connected:
            return True
        try:
            with self._lock:
                self.rpc = Presence(self.client_id)
                self.rpc.connect()
                self.connected = True
                return True
        except Exception:
            self.connected = False
            return False

    def clear_presence(self) -> None:
        if self.rpc and self.connected:
            try:
                with self._lock:
                    self.rpc.clear()
            except Exception:
                pass

    def disconnect(self) -> None:
        if self.rpc and self.connected:
            try:
                with self._lock:
                    self.rpc.close()
                    self.connected = False
            except Exception:
                pass

    def _update(
        self,
        details: Optional[str] = None,
        state: Optional[str] = None,
        large_image: Optional[str] = None,
        large_text: Optional[str] = None,
        small_image: Optional[str] = None,
        small_text: Optional[str] = None,
        buttons: bool = True,
    ) -> None:
        if not self.connected or not self.rpc:
            return
        try:
            with self._lock:
                buttons_list = [
                    {"label": "Discord Server", "url": "https://discord.gg/asJkqwqb"},
                    {"label": "Website", "url": "https://amverge.app/"},
                ] if buttons else None
                self.rpc.update(
                    details=details,
                    state=state,
                    large_image=large_image or "amverge_logo",
                    large_text=large_text or "AMVerge",
                    small_image=small_image,
                    small_text=small_text,
                    buttons=buttons_list,
                )
        except Exception:
            self.connected = False

    def update_idle(self) -> None:
        self._update(details="Ready to process videos", state="Idle")

    def update_detecting(self, file_name: str = "", progress: float = 0) -> None:
        self._update(
            details=f"File: {file_name}" if file_name else "Processing video",
            state=f"Detecting Scenes ({progress:.0f}%)",
            small_image="loading_icon_new",
            small_text="Detecting...",
        )

    def update_selecting(self, count: int = 0) -> None:
        self._update(
            details="Editing Episode",
            state=f"Selecting Clips ({count} selected)",
        )

    def update_navigating(self, page: str = "") -> None:
        self._update(
            details="Navigating menus",
            state=f"In {page.capitalize()}" if page else "Browsing",
        )

    def update_exporting(self, file_name: str = "", progress: float = 0) -> None:
        self._update(
            details=f"Saving: {file_name}" if file_name else "Exporting clips",
            state=f"Exporting ({progress:.0f}%)",
            small_image="save_icon_new",
            small_text="Exporting...",
        )

    def update_merging(self) -> None:
        self._update(
            details="Merging clips",
            state="Merging",
            small_image="save_icon_new",
            small_text="Merging...",
        )

    def update_complete(self) -> None:
        self._update(
            details="Process complete",
            state="Done",
            small_image="check_icon_new",
            small_text="Done",
        )

    def update_error(self, msg: str = "") -> None:
        self._update(
            details=msg[:128] if msg else "An error occurred",
            state="Error",
            small_image="edit_icon_new",
            small_text="Error",
        )

    def __enter__(self) -> "DiscordRPC":
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.clear_presence()
        self.disconnect()


_instance: Optional[DiscordRPC] = None


def get_rpc() -> DiscordRPC:
    global _instance
    if _instance is None:
        _instance = DiscordRPC()
    return _instance
