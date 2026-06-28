import typer

from .commands.detect import detect
from .commands.export import export
from .commands.merge import merge
from .commands.info import info
from .commands.about import about
from .commands.credits import credits
from .commands.changelog import changelog
from .commands.usage import usage
from .commands.backend import backend
from .commands.rpc_server import rpc_server

app = typer.Typer(
    name="amverge",
    help="AMVerge CLI - scene detection and clip management.",
    no_args_is_help=False,
    pretty_exceptions_show_locals=False,
)

# Workflow
app.command()(detect)
app.command()(export)
app.command()(merge)
app.command()(info)

# App backend replacement (hidden - called by Rust sidecar)
app.command(hidden=True)(backend)
app.command(name="rpc-server", hidden=True)(rpc_server)

# Info
app.command()(usage)
app.command()(about)
app.command()(credits)
app.command()(changelog)


@app.callback(invoke_without_command=True)
def _default(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        from .wizard import run_wizard
        run_wizard()


if __name__ == "__main__":
    app()
