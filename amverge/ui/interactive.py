from __future__ import annotations

import os
import sys
from typing import Optional

from rich.console import Console
from rich.text import Text


def _getch():
    if os.name == "nt":
        import msvcrt
        return msvcrt.getch()
    else:
        import termios
        import tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _read_key():
    ch = _getch()
    if ch == b"\xe0" or ch == b"\x00":
        ch = _getch()
        if ch == b"H":
            return "up"
        elif ch == b"P":
            return "down"
        elif ch == b"M":
            return "right"
        elif ch == b"K":
            return "left"
        return None
    if ch == b"\r" or ch == b"\n":
        return "enter"
    if ch == b" ":
        return "space"
    if ch == b"\x1b":
        try:
            nxt = _getch()
            if nxt == b"[":
                arrow = _getch()
                if arrow == b"A":
                    return "up"
                elif arrow == b"B":
                    return "down"
                elif arrow == b"C":
                    return "right"
                elif arrow == b"D":
                    return "left"
                return None
            return "escape"
        except Exception:
            return "escape"
    if ch == b"q" or ch == b"Q":
        return "quit"
    return None


def select(
    options: list[str],
    message: str = "Select:",
    default_index: int = 0,
    console: Optional[Console] = None,
) -> int:
    if console is None:
        from amverge.ui import err as _c
        console = _c

    idx = default_index
    n = len(options)

    def _render():
        lines = [f"  [accent]>[/]  [label]{message}[/]"]
        for i, opt in enumerate(options):
            prefix = " [accent bold]>[/]" if i == idx else "   "
            style = "[accent]" if i == idx else "[dim]"
            lines.append(f"  {prefix}  {style}{opt}[/]")
        lines.append("")
        lines.append("  [muted]  arrow keys to move, enter to select, q to cancel[/]")
        return "\n".join(lines)

    render = _render()
    console.print(render)

    while True:
        key = _read_key()
        if key == "up":
            idx = (idx - 1) % n
        elif key == "down":
            idx = (idx + 1) % n
        elif key == "enter":
            console.print(f"  [accent]>[/]  [label]{message}[/] [accent]{options[idx]}[/]")
            return idx
        elif key == "quit" or key == "escape":
            return -1
        else:
            continue

        console.clear()
        render = _render()
        console.print(render)


def checkboxes(
    options: list[str],
    message: str = "Select:",
    defaults: Optional[list[int]] = None,
    console: Optional[Console] = None,
) -> list[int]:
    if console is None:
        from amverge.ui import err as _c
        console = _c

    if defaults is None:
        defaults = []
    selected = set(defaults)
    idx = 0
    n = len(options)

    def _render():
        lines = [f"  [accent]>[/]  [label]{message}[/]"]
        for i, opt in enumerate(options):
            mark = "[accent bold][x][/]" if i in selected else "[dim][ ][/]"
            cursor = " [accent bold]>[/]" if i == idx else "   "
            style = "[accent]" if i == idx else ""
            name = f"{style}{opt}[/]" if style else f"[dim]{opt}[/]"
            lines.append(f"  {cursor}  {mark}  {name}")
        lines.append("")
        lines.append("  [muted]  arrow keys to move, space to toggle, enter to confirm, q to cancel[/]")
        return "\n".join(lines)

    render = _render()
    console.print(render)

    while True:
        key = _read_key()
        if key == "up":
            idx = (idx - 1) % n
        elif key == "down":
            idx = (idx + 1) % n
        elif key == "space":
            if idx in selected:
                selected.discard(idx)
            else:
                selected.add(idx)
        elif key == "enter":
            if not selected:
                selected.add(idx)
            result = sorted(selected)
            names = ", ".join(options[i] for i in result)
            console.print(f"  [accent]>[/]  [label]{message}[/] [accent]{names}[/]")
            return result
        elif key == "quit" or key == "escape":
            return []
        else:
            continue

        console.clear()
        render = _render()
        console.print(render)


def confirm(message: str = "Confirm?", default: bool = True, console: Optional[Console] = None) -> bool:
    if console is None:
        from amverge.ui import err as _c
        console = _c

    options = ["Yes", "No"]
    default_idx = 0 if default else 1
    result = select(options, message, default_idx, console)
    return result == 0


def text_input(message: str = "Input:", default: str = "", console: Optional[Console] = None) -> str:
    if console is None:
        from amverge.ui import err as _c
        console = _c

    hint = f" [muted][{default}][/]" if default else ""
    console.print(f"  [accent]>[/]  [label]{message}[/]{hint}")

    val = input("  > ").strip()
    return val if val else default
