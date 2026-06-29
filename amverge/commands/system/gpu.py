from __future__ import annotations

from ...ui import banner, console, make_table


def gpu() -> None:
    """Show PyTorch, CUDA, and GPU diagnostics."""
    banner("gpu")

    t = make_table(
        ("", "muted",  {"width": 22, "no_wrap": True}),
        ("", "label",  {}),
        title="PyTorch",
    )

    try:
        import torch
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()

        t.add_row("torch version",   torch_version)
        t.add_row("CUDA available",  "[accent]yes[/]" if cuda_available else "[warn]no[/]")

        if cuda_available:
            t.add_row("CUDA version",    torch.version.cuda or "-")
            device_count = torch.cuda.device_count()
            t.add_row("GPU count",       str(device_count))
            for i in range(device_count):
                name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_memory / (1024 ** 3)
                t.add_row(f"GPU {i}",    f"{name}  ({vram_gb:.1f} GB VRAM)")
        else:
            t.add_row("GPU",             "[muted]none - will use CPU[/]")

    except ImportError:
        t.add_row("torch",           "[error]not installed[/]")
        t.add_row("install",         "pip install amverge[ml]")

    console.print(t)

    t2 = make_table(
        ("", "muted",  {"width": 22, "no_wrap": True}),
        ("", "label",  {}),
        title="ML Dependencies",
    )

    try:
        from transnetv2_pytorch import TransNetV2  # noqa: F401
        t2.add_row("transnetv2-pytorch", "[accent]installed[/]")
    except ImportError:
        t2.add_row("transnetv2-pytorch", "[error]not installed[/]  pip install amverge[ml]")

    try:
        import tqdm  # noqa: F401
        t2.add_row("tqdm",               f"[accent]installed[/]  v{tqdm.__version__}")
    except ImportError:
        t2.add_row("tqdm",               "[error]not installed[/]  pip install amverge[ml]")

    try:
        from ...core.detection.nelux_runtime import _get_nelux_video_reader
        _get_nelux_video_reader()
        t2.add_row("nelux",              "[accent]available[/]")
    except ImportError as e:
        if "Failed to import nelux" in str(e):
            t2.add_row("nelux",          "[warn]DLLs not found[/]  set AMVERGE_FFMPEG_BIN")
        else:
            t2.add_row("nelux",          "[muted]not installed[/]  (optional, Windows only)")
    except Exception:
        t2.add_row("nelux",              "[muted]not installed[/]  (optional, Windows only)")

    console.print(t2)

    t3 = make_table(
        ("", "muted",  {"width": 22, "no_wrap": True}),
        ("", "label",  {}),
        title="Optional Extras",
    )

    try:
        import cv2  # noqa: F401
        t3.add_row("opencv (edge)",      f"[accent]installed[/]  v{cv2.__version__}")
    except ImportError:
        t3.add_row("opencv (edge)",      "[muted]not installed[/]  pip install amverge[edge]")

    try:
        import pypresence  # noqa: F401
        t3.add_row("pypresence (RPC)",   "[accent]installed[/]")
    except ImportError:
        t3.add_row("pypresence (RPC)",   "[muted]not installed[/]  pip install amverge[discord]")

    console.print(t3)
