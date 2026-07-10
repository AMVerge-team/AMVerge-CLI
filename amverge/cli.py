import typer

from .commands.detection.detect import detect
from .commands.export.export import export
from .commands.export.merge import merge
from .commands.info.info import info
from .commands.info.probe import probe
from .commands.system.gpu import gpu
from .commands.system.doctor import doctor
from .commands.system.version import version
from .commands.detection.bench import bench
from .commands.detection.cache import cache
from .commands.detection.keyframes import keyframes
from .commands.detection.scenes import scenes
from .commands.about.about import about
from .commands.about.credits import credits
from .commands.about.changelog import changelog, whatsnew
from .commands.about.usage import usage
from .commands.upscaling.upscale import upscale
from .commands.upscaling.models import models
from .commands.interpolation.flowframes import flowframes
from .commands.interpolation.flowframes_path import flowframes_path as flowframes_path_cmd
from .commands.interpolation.interpolate import interpolate
from .commands.sidecar.backend import backend
from .commands.sidecar.rpc_server import rpc_server

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
app.command()(probe)
app.command()(gpu)
app.command()(doctor)
app.command()(version)
app.command()(bench)
app.command()(cache)
app.command()(keyframes)
app.command()(scenes)

# App backend replacement (hidden - called by Rust sidecar)
app.command(hidden=True)(backend)
app.command(name="rpc-server", hidden=True)(rpc_server)

# Upscale
app.command()(upscale)
app.command()(models)

# Interpolation
app.command()(interpolate)
app.command(name="flowframes")(flowframes)
app.command(name="flowframes-path")(flowframes_path_cmd)

# Info
app.command()(usage)
app.command()(about)
app.command()(credits)
app.command()(changelog)
app.command(name="whatsnew")(whatsnew)


@app.callback(invoke_without_command=True)
def _default(ctx: typer.Context) -> None:
    if ctx.invoked_subcommand is None:
        from .wizard import run_wizard
        run_wizard()


if __name__ == "__main__":
    app()
