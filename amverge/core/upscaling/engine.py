import gc
import os
import subprocess
import threading
from pathlib import Path
from typing import Callable, Optional

from .anime4k import upscale_video_anime4k
from .artcnn import upscale_video_artcnn
from .ffmpeg_helpers import CREATE_NO_WINDOW, build_ffmpeg_pipe, get_color_args, mux_audio
from .monitor import SystemMonitor
from .registry import QUALITY_PRESETS, get_model

UPSCALE_AVAILABLE = False
try:
    import torch
    import cv2
    UPSCALE_AVAILABLE = True
except ImportError:
    pass

SPANDREL_AVAILABLE = False
try:
    import spandrel
    SPANDREL_AVAILABLE = True
except ImportError:
    pass


def _get_dispatch_info(model_key):
    entry = get_model(model_key)
    if entry is None:
        raise ValueError(f"Unknown model key: {model_key}")
    method = entry["method"]
    scales = entry["scales"]
    name = entry.get("name", model_key)
    credit = entry.get("credit", "")
    return method, scales, name, credit, entry


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _tensor_to_frame(tensor, device="cuda"):
    frame = tensor.squeeze(0).permute(1, 2, 0).detach()
    frame = torch.clamp(frame, 0, 1)
    frame = (frame * 255).contiguous().to(torch.uint8)
    if device == "cuda" or str(frame.device).startswith("cuda"):
        frame = frame.cpu()
    return cv2.cvtColor(frame.numpy(), cv2.COLOR_RGB2BGR)


def _frame_to_tensor(frame, device):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(frame_rgb).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def _upscale_ml(input_path, output_path, model_key, entry, scale, preset, fit_w, fit_h, progress_cb):
    q = QUALITY_PRESETS.get(preset, QUALITY_PRESETS["high"])
    device = _get_device()

    cpu_threads = max(2, min(os.cpu_count() or 4, 8))
    cv2.setNumThreads(cpu_threads)

    from .weight_loader import is_weight_downloaded, download_weights, get_weight_path, verify_weight_hash

    if not is_weight_downloaded(model_key):
        download_weights(model_key)

    weight_path = get_weight_path(model_key)
    if entry.get("hash"):
        verify_weight_hash(model_key, weight_path)

    if not SPANDREL_AVAILABLE:
        raise RuntimeError("spandrel required. Run: pip install amverge[upscale]")

    model = spandrel.ModelLoader(device=device).load_from_file(weight_path)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open: {input_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_val = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    out_w = w * scale
    out_h = h * scale

    extra_vf = None
    if fit_w > 0 and fit_h > 0 and (out_w > fit_w or out_h > fit_h):
        extra_vf = f"scale={fit_w}:{fit_h}:force_original_aspect_ratio=decrease:flags=lanczos"

    ffmpeg_cmd = build_ffmpeg_pipe(out_w, out_h, fps_val, q["crf"], q["x264"], q.get("tune"), output_path, extra_vf,
                                   color_args=get_color_args(input_path))

    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                                   creationflags=CREATE_NO_WINDOW)

    def _read_stderr():
        try:
            for _ in ffmpeg_proc.stderr:
                pass
        except Exception:
            pass
    threading.Thread(target=_read_stderr, daemon=True).start()

    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            tensor = _frame_to_tensor(frame, device)
            with torch.no_grad():
                upscaled = model(tensor)
            del tensor
            result = _tensor_to_frame(upscaled, str(device))
            del upscaled
            if ffmpeg_proc.poll() is None:
                try:
                    ffmpeg_proc.stdin.write(result.tobytes())
                except (BrokenPipeError, OSError):
                    break
            del result
            frame_idx += 1
            if frame_idx % 5 == 0:
                gc.collect()
            if device.type == "cuda" and frame_idx % 10 == 0:
                torch.cuda.empty_cache()
            if progress_cb:
                pct = min(100, int((frame_idx / total_frames) * 100))
                progress_cb(pct, f"Upscaling... {frame_idx}/{total_frames}")
    finally:
        cap.release()
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        try:
            ffmpeg_proc.stdin.close()
        except Exception:
            pass
        ffmpeg_proc.wait()

    mux_audio(str(output_path), str(input_path))


def upscale_model(
    model_key: str,
    input_path: str | Path,
    output_path: str | Path,
    scale: int = 2,
    preset: str = "high",
    fit_w: int = 0,
    fit_h: int = 0,
    mode: Optional[str] = None,
    progress_cb: Optional[Callable[[int, str], None]] = None,
    monitor: Optional[SystemMonitor] = None,
) -> None:
    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    method, scales, name, credit, entry = _get_dispatch_info(model_key)

    if scale not in scales:
        scales_str = "/".join(f"{s}x" for s in scales)
        raise ValueError(f"Model '{model_key}' supports {scales_str}, got {scale}x")

    if monitor:
        monitor.start()

    if method == "ml":
        if not UPSCALE_AVAILABLE:
            raise RuntimeError("ML upscaling requires torch and opencv. Run: pip install amverge[upscale]")
        _upscale_ml(input_path, output_path, model_key, entry, scale, preset, fit_w, fit_h, progress_cb)

    elif method == "shader":
        if mode:
            entry = dict(entry)
            entry["default_mode"] = mode
        upscale_video_anime4k(input_path, output_path, entry, scale, preset, fit_w, fit_h, progress_cb)

    elif method == "onnx":
        upscale_video_artcnn(input_path, output_path, model_key, entry, scale, preset, fit_w, fit_h, progress_cb)

    else:
        raise ValueError(f"Unknown method '{method}' for model '{model_key}'")

    if monitor:
        monitor.stop()
