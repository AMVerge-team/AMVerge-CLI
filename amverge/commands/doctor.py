from __future__ import annotations

from ..core.diagnostics import check_environment
from ..ui import banner, console, make_table


def doctor() -> None:
    """Run a full environment health check."""
    banner("doctor")

    result = check_environment()

    t = make_table(
        ("", "muted",  {"width": 26, "no_wrap": True}),
        ("", "label",  {"width": 3,  "no_wrap": True}),
        ("", "muted",  {}),
        title=f"Health Check  {result.passed}/{result.total} passed",
    )

    for c in result.checks:
        status = "[accent]pass[/]" if c.ok else "[error]FAIL[/]"
        note = c.detail if c.ok else (f"[error]{c.detail}[/]" + (f"  [muted]{c.fix}[/]" if c.fix else ""))
        t.add_row(c.label, status, note)

    console.print(t)

    if result.is_healthy:
        console.print("[accent]All checks passed.[/]\n")
    else:
        console.print(f"[error]{result.failed} check(s) failed.[/]  See fix hints above.\n")
