from __future__ import annotations

from rich.markup import escape

from ...ui import banner, console, make_table


def gpu() -> None:
    """Show PyTorch, CUDA, and GPU diagnostics."""
    banner("gpu")

    from ...core.infra.device import detect_gpu, torch_backend_gap

    detected = detect_gpu()

    t0 = make_table(
        ("", "muted",  {"width": 22, "no_wrap": True}),
        ("", "label",  {}),
        title="Detected GPU",
    )
    if detected.available:
        t0.add_row("vendor",         f"[accent]{detected.vendor}[/]")
        t0.add_row("name",           escape(detected.name or "-"))
        t0.add_row("VRAM",           f"{detected.vram_gb:.1f} GB" if detected.vram_gb else "[muted]unknown[/]")
        t0.add_row("driver",         detected.driver or "[muted]unknown[/]")
    else:
        t0.add_row("vendor",         "[warn]none detected[/]")
    console.print(t0)

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
                t.add_row(f"GPU {i}",    f"{escape(name)}  ({vram_gb:.1f} GB VRAM)")
        else:
            t.add_row("device",          "[warn]cpu[/]")

    except ImportError:
        t.add_row("torch",           "[error]not installed[/]")
        t.add_row("install",         escape(f"{escape('pip install amverge[ml]')}"))

    console.print(t)

    t1 = make_table(
        ("", "muted",  {"width": 22, "no_wrap": True}),
        ("", "label",  {}),
        title="GPU Backends",
    )

    gap = torch_backend_gap()
    if gap is None:
        t1.add_row("PyTorch (CUDA)",     "[accent]active[/]  ml upscaling, RIFE, TransNetV2, depth")
    elif gap == "torch_not_cuda":
        t1.add_row("PyTorch (CUDA)",     "[warn]cpu only[/]  torch built without CUDA")
    elif gap == "no_torch_backend":
        t1.add_row("PyTorch (CUDA)",     f"[warn]cpu only[/]  no CUDA backend for a {detected.vendor} GPU")
    else:
        t1.add_row("PyTorch (CUDA)",     "[warn]cpu only[/]  no GPU detected")

    try:
        import onnxruntime
        from ...core.upscaling.artcnn import resolve_onnx_providers
        provs = resolve_onnx_providers(onnxruntime)
        if "DmlExecutionProvider" in provs:
            t1.add_row("ONNX (DirectML)", "[accent]active[/]  ArtCNN on any vendor")
        elif "CUDAExecutionProvider" in provs:
            t1.add_row("ONNX (DirectML)", "[muted]not needed[/]  CUDA provider active")
        elif detected.available and not detected.is_nvidia:
            t1.add_row("ONNX (DirectML)", f"[warn]absent[/]  {escape('pip install amverge[upscale-amd]')}")
        else:
            t1.add_row("ONNX (CPU)",      "[muted]cpu only[/]")
    except ImportError:
        t1.add_row("ONNX",               f"[muted]not installed[/]  {escape('pip install amverge[upscale]')}")

    try:
        from ...core.upscaling.anime4k import libplacebo_available
        if libplacebo_available():
            t1.add_row("Vulkan (libplacebo)", "[accent]active[/]  Anime4K on any vendor")
        else:
            t1.add_row("Vulkan (libplacebo)", "[warn]absent[/]  ffmpeg lacks libplacebo, Anime4K falls back to lanczos")
    except Exception:
        t1.add_row("Vulkan (libplacebo)", "[muted]unknown[/]")

    try:
        from ...core.interpolation.flowframes import flowframes_available
        if flowframes_available():
            t1.add_row("Vulkan (Flowframes)", "[accent]active[/]  RIFE ncnn on any vendor")
        else:
            t1.add_row("Vulkan (Flowframes)", "[muted]absent[/]  set it up with 'amverge flowframes-path'")
    except Exception:
        t1.add_row("Vulkan (Flowframes)", "[muted]unknown[/]")

    console.print(t1)

    t2 = make_table(
        ("", "muted",  {"width": 22, "no_wrap": True}),
        ("", "label",  {}),
        title="ML Dependencies",
    )

    try:
        from transnetv2_pytorch import TransNetV2  # noqa: F401
        t2.add_row("transnetv2-pytorch", "[accent]installed[/]")
    except ImportError:
        t2.add_row("transnetv2-pytorch", f"[error]not installed[/]  {escape('pip install amverge[ml]')}")

    try:
        import tqdm  # noqa: F401
        t2.add_row("tqdm",               f"[accent]installed[/]  v{tqdm.__version__}")
    except ImportError:
        t2.add_row("tqdm",               f"[error]not installed[/]  {escape('pip install amverge[ml]')}")

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
        t3.add_row("opencv (edge)",      f"[muted]not installed[/]  {escape('pip install amverge[edge]')}")

    try:
        import pypresence  # noqa: F401
        t3.add_row("pypresence (RPC)",   "[accent]installed[/]")
    except ImportError:
        t3.add_row("pypresence (RPC)",   f"[muted]not installed[/]  {escape('pip install amverge[discord]')}")

    try:
        from ...core.depth import DEPTH_AVAILABLE
        if DEPTH_AVAILABLE:
            t3.add_row("depth-anything-v2", "[accent]installed[/]")
        else:
            t3.add_row("depth-anything-v2", f"[muted]not installed[/]  {escape('pip install amverge[depth]')}")
    except ImportError:
        t3.add_row("depth-anything-v2", f"[muted]not installed[/]  {escape('pip install amverge[depth]')}")

    try:
        from ...core.upscaling import UPSCALE_AVAILABLE
        if UPSCALE_AVAILABLE:
            t3.add_row("upscale", "[accent]installed[/]")
        else:
            t3.add_row("upscale", f"[muted]not installed[/]  {escape('pip install amverge[upscale]')}")
    except ImportError:
        t3.add_row("upscale", f"[muted]not installed[/]  {escape('pip install amverge[upscale]')}")

    try:
        from ...core.interpolation import INTERPOLATION_AVAILABLE
        if INTERPOLATION_AVAILABLE:
            t3.add_row("interpolation", "[accent]installed[/]")
        else:
            t3.add_row("interpolation", f"[muted]not installed[/]  {escape('pip install amverge[interpolation]')}")
    except ImportError:
        t3.add_row("interpolation", f"[muted]not installed[/]  {escape('pip install amverge[interpolation]')}")

    console.print(t3)
