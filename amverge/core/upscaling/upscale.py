import gc
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..infra.binaries import get_ffmpeg, get_ffprobe
from .registry import QUALITY_PRESETS

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

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

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import warnings
warnings.filterwarnings("ignore", message=".*expandable_segments.*")


def _resolve_quality(key):
    return QUALITY_PRESETS.get(str(key or "high").lower(), QUALITY_PRESETS["high"])


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


def _load_model(model_name, scale, device):
    from .weight_loader import (
        is_weight_downloaded, download_weights,
        get_weight_path, verify_weight_hash, load_weights_if_available,
    )
    from .shufflecugan import ShuffleCUGANModel

    if not is_weight_downloaded(model_name):
        success = download_weights(model_name)
        if not success:
            raise RuntimeError(f"Failed to download weights for {model_name}")

    weight_path = None
    try:
        weight_path = get_weight_path(model_name)
        if weight_path and os.path.exists(weight_path):
            verify_weight_hash(model_name, weight_path)
    except ValueError:
        pass

    try:
        model = ShuffleCUGANModel(model_name, scale).to(device)
        model.eval()
        if weight_path and os.path.exists(weight_path):
            if load_weights_if_available(model, model_name, device):
                return model
    except Exception:
        pass

    if not SPANDREL_AVAILABLE:
        raise RuntimeError(
            "Cannot load upscale model. spandrel required but not installed. "
            "Run: pip install amverge[upscale]"
        )

    try:
        if weight_path and os.path.exists(weight_path):
            model_descriptor = spandrel.ModelLoader(device=device).load_from_file(weight_path)
            return model_descriptor
    except Exception as e:
        raise RuntimeError(f"spandrel loading failed: {e}")

    raise RuntimeError(f"Weights not available for {model_name}")


def _build_ffmpeg_pipe_cmd(input_path, output_path, output_w, output_h, output_fps,
                            scale, crf, x264_preset, tune, max_w=0, max_h=0):
    out_w = output_w * scale if scale > 1 else output_w
    out_h = output_h * scale if scale > 1 else output_h

    ffmpeg = get_ffmpeg()
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}", "-pix_fmt", "bgr24",
        "-r", str(output_fps), "-i", "-",
        "-c:v", "libx264", "-crf", str(crf), "-preset", x264_preset,
        "-profile:v", "high", "-level:v", "5.1",
    ]
    if tune:
        cmd += ["-tune", tune]
    if max_w > 0 and max_h > 0 and (out_w > max_w or out_h > max_h):
        cmd += ["-vf", f"scale={max_w}:{max_h}:force_original_aspect_ratio=decrease:flags=lanczos"]
    cpu_cores = os.cpu_count() or 4
    enc_threads = max(2, min(cpu_cores, 4 if out_w * out_h >= 3840 * 2160 else 8))
    cmd += [
        "-pix_fmt", "yuv420p",
        "-x264-params", f"threads={enc_threads}:lookahead-threads=1:rc-lookahead=20",
        "-threads", str(enc_threads),
        "-movflags", "+faststart",
        str(output_path),
    ]
    return cmd


def _mux_audio(video_path, audio_source_path):
    ffprobe = get_ffprobe()
    probe_cmd = [
        ffprobe, "-v", "error", "-select_streams", "a", "-show_entries",
        "stream=codec_type", "-of", "csv=p=0", str(audio_source_path),
    ]
    has_audio = False
    try:
        r = subprocess.run(probe_cmd, capture_output=True, text=True,
                           timeout=10, creationflags=CREATE_NO_WINDOW)
        if "audio" in r.stdout.lower():
            has_audio = True
    except Exception:
        pass

    if not has_audio:
        return False

    ffmpeg = get_ffmpeg()
    tmp = str(video_path) + ".tmp.mp4"
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video_path),
        "-i", str(audio_source_path),
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-movflags", "+faststart",
        tmp,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=300, creationflags=CREATE_NO_WINDOW)
        if r.returncode == 0:
            os.replace(tmp, str(video_path))
            return True
        if os.path.exists(tmp):
            os.unlink(tmp)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
    return False


def upscale_video(
    input_path: str | Path,
    output_path: str | Path,
    model_name: str = "adore",
    scale: int = 2,
    preset: str = "high",
    fit_w: int = 0,
    fit_h: int = 0,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not UPSCALE_AVAILABLE:
        raise RuntimeError(
            "Upscaling requires torch and opencv. Run: pip install amverge[upscale]"
        )

    q = _resolve_quality(preset)
    device = _get_device()

    cpu_threads = max(2, min(os.cpu_count() or 4, 8))
    cv2.setNumThreads(cpu_threads)

    model = _load_model(model_name, scale, device)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 1

    out_w = w * scale
    out_h = h * scale
    ffmpeg_cmd = _build_ffmpeg_pipe_cmd(
        input_path, output_path, w, h, fps, scale,
        q["crf"], q["x264"], q["tune"], fit_w, fit_h,
    )

    ffmpeg_proc = subprocess.Popen(
        ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
        creationflags=CREATE_NO_WINDOW,
    )

    stderr_lines = []
    def _read_stderr():
        try:
            for line in ffmpeg_proc.stderr:
                decoded = line.decode(errors="replace").strip()
                if decoded:
                    stderr_lines.append(decoded)
        except Exception:
            pass
    stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
    stderr_thread.start()

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
                denom = max(total_frames, 1)
                pct = min(100, int((frame_idx / denom) * 100))
                progress_cb(pct, f"Upscaling frames... {frame_idx}/{denom}")

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
        if stderr_lines:
            stderr_summary = " | ".join(stderr_lines[-5:])

    _mux_audio(str(output_path), str(input_path))
