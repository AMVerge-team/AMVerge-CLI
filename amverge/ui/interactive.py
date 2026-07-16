from __future__ import annotations

from typing import Optional

from rich.console import Console

try:
    import questionary
    from questionary import Style

    INTERACTIVE_AVAILABLE = True
except ImportError:
    questionary = None
    Style = None
    INTERACTIVE_AVAILABLE = False

_AMVERGE_STYLE = None


def _check():
    if not INTERACTIVE_AVAILABLE:
        raise ImportError(
            "questionary is required for interactive mode. Install with: pip install questionary"
        )


def _style():
    global _AMVERGE_STYLE
    if _AMVERGE_STYLE is None:
        _check()
        _AMVERGE_STYLE = Style([
            ("qmark", "fg:#22c55e bold"),
            ("question", "bold"),
            ("answer", "fg:#22c55e bold"),
            ("pointer", "fg:#22c55e bold"),
            ("highlighted", "fg:#22c55e bold"),
            ("selected", "fg:#22c55e"),
            ("instruction", "fg:#5f5f5f"),
            ("text", ""),
        ])
    return _AMVERGE_STYLE


def select(
    options: list[str],
    message: str = "Select:",
    default_index: int = 0,
    console: Optional[Console] = None,
) -> int:
    _check()
    default_choice = options[default_index] if 0 <= default_index < len(options) else None

    answer = questionary.select(
        message,
        choices=options,
        default=default_choice,
        style=_style(),
        qmark=">",
    ).ask()

    if answer is None:
        return -1
    return options.index(answer)


def checkboxes(
    options: list[str],
    message: str = "Select:",
    defaults: Optional[list[int]] = None,
    console: Optional[Console] = None,
) -> list[int]:
    _check()
    defaults = defaults or []

    choices = [
        questionary.Choice(title=opt, checked=(i in defaults))
        for i, opt in enumerate(options)
    ]

    answer = questionary.checkbox(
        message,
        choices=choices,
        style=_style(),
        qmark=">",
        instruction="(space to toggle, enter to confirm)",
    ).ask()

    if answer is None:
        return []
    return [options.index(a) for a in answer]


def confirm(message: str = "Confirm?", default: bool = True, console: Optional[Console] = None) -> bool:
    _check()
    options = ["Yes", "No"]
    default_choice = "Yes" if default else "No"

    answer = questionary.select(
        message,
        choices=options,
        default=default_choice,
        style=_style(),
        qmark=">",
    ).ask()

    if answer is None:
        return False
    return answer == "Yes"


def text_input(message: str = "Input:", default: str = "", console: Optional[Console] = None) -> str:
    _check()
    answer = questionary.text(
        message,
        default=default,
        style=_style(),
        qmark=">",
    ).ask()
    return answer if answer is not None else ""
