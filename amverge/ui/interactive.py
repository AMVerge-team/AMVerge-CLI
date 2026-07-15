from __future__ import annotations

import os
import sys
from typing import Optional

from rich.console import Console


def _getch():
    if os.name == "nt":
        import msvcrt
        ch = msvcrt.getch()
        if ch == b"\xe0" or ch == b"\x00":
            ch += msvcrt.getch()
        return ch
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
    if isinstance(ch, bytes) and len(ch) == 2 and ch[0:1] in (b"\xe0", b"\x00"):
        second = ch[1:2]
        if second == b"H": return "up"
        if second == b"P": return "down"
        return None
    if ch in (b"\r", b"\n"):
        return "enter"
    if ch == b" ":
        return "space"
    if ch == b"\x1b":
        try:
            nxt = _getch()
            if nxt == b"[":
                arrow = _getch()
                if arrow == b"A": return "up"
                if arrow == b"B": return "down"
                return None
            return "escape"
        except Exception:
            return "escape"
    if ch in (b"q", b"Q"):
        return "quit"
    return None


def _hide_cursor():
    sys.stdout.write("\033[?25l")
    sys.stdout.flush()


def _show_cursor():
    sys.stdout.write("\033[?25h")
    sys.stdout.flush()


def _ensure_ansi():
    if os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            pass


_ensure_ansi()


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
    _hide_cursor()

    def _render():
        lines = [f"  [accent]>[/]  [label]{message}[/]"]
        for i, opt in enumerate(options):
            prefix = " [accent bold]>[/]" if i == idx else "   "
            style = "[accent]" if i == idx else "[dim]"
            lines.append(f"  {prefix}  {style}{opt}[/]")
        lines.append("")
        lines.append("  [muted]  arrows: move  |  enter: select  |  q: cancel[/]")
        return lines

    prev_count = 0
    try:
        while True:
            current_lines = _render()
            if prev_count > 0:
                sys.stdout.write(f"\033[{prev_count}A")
            line_count = len(current_lines)
            for i, line in enumerate(current_lines):
                sys.stdout.write("\033[K")
                console.print(line)
                sys.stdout.write("\n")
            sys.stdout.flush()
            prev_count = line_count

            key = _read_key()
            if key == "up":
                idx = (idx - 1) % n
            elif key == "down":
                idx = (idx + 1) % n
            elif key == "enter":
                sys.stdout.write(f"\033[{prev_count}A")
                for __ in range(prev_count):
                    sys.stdout.write("\033[K\n")
                sys.stdout.write(f"\033[{prev_count}A")
                sys.stdout.flush()
                console.print(f"  [accent]>[/]  [label]{message}[/] [accent]{options[idx]}[/]")
                return idx
            elif key in ("quit", "escape"):
                sys.stdout.write(f"\033[{prev_count}A")
                for __ in range(prev_count):
                    sys.stdout.write("\033[K\n")
                sys.stdout.write(f"\033[{prev_count}A")
                sys.stdout.flush()
                return -1
    finally:
        _show_cursor()


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
    _hide_cursor()

    def _render():
        lines = [f"  [accent]>[/]  [label]{message}[/]"]
        for i, opt in enumerate(options):
            mark = "[accent bold][x][/]" if i in selected else "[dim][ ][/]"
            cursor = " [accent bold]>[/]" if i == idx else "   "
            style = "[accent]" if i == idx else ""
            name = f"{style}{opt}[/]" if style else f"[dim]{opt}[/]"
            lines.append(f"  {cursor}  {mark}  {name}")
        lines.append("")
        lines.append("  [muted]  arrows: move  |  space: toggle  |  enter: confirm  |  q: cancel[/]")
        return lines

    prev_count = 0
    try:
        while True:
            current_lines = _render()
            if prev_count > 0:
                sys.stdout.write(f"\033[{prev_count}A")
            line_count = len(current_lines)
            for line in current_lines:
                sys.stdout.write("\033[K")
                console.print(line)
                sys.stdout.write("\n")
            sys.stdout.flush()
            prev_count = line_count

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
                sys.stdout.write(f"\033[{prev_count}A")
                for __ in range(prev_count):
                    sys.stdout.write("\033[K\n")
                sys.stdout.write(f"\033[{prev_count}A")
                sys.stdout.flush()
                console.print(f"  [accent]>[/]  [label]{message}[/] [accent]{names}[/]")
                return result
            elif key in ("quit", "escape"):
                sys.stdout.write(f"\033[{prev_count}A")
                for __ in range(prev_count):
                    sys.stdout.write("\033[K\n")
                sys.stdout.write(f"\033[{prev_count}A")
                sys.stdout.flush()
                return []
    finally:
        _show_cursor()


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
