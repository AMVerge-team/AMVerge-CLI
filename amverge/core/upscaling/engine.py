import gc
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..infra.binaries import get_ffmpeg, get_ffprobe
from ..infra.config import get_models_dir
from .registry import UPSCALE_REGISTRY, QUALITY_PRESETS, get_model

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

ANIME4K_MODE_PRESETS = {
    "light": ["Anime4K_Upscale_CNN_x2_VL.glsl"],
    "medium": ["Anime4K_DarkLines_VeryFast.glsl", "Anime4K_ThinLines_VeryFast.glsl", "Anime4K_Upscale_CNN_x2_M.glsl"],
    "strong": ["Anime4K_Clamp_Highlights.glsl", "Anime4K_Restore_CNN_VL.glsl", "Anime4K_Upscale_CNN_x2_VL.glsl"],
}


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
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0?",
        "-movflags", "+faststart", tmp,
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


def _get_dispatch_info(model_key):
    entry = get_model(model_key)
    if entry is None:
        raise ValueError(f"Unknown model key: {model_key}")
    method = entry["method"]
    scales = entry["scales"]
    name = entry.get("name", model_key)
    credit = entry.get("credit", "")
    return method, scales, name, credit, entry


def _ensure_ffmpeg():
    from ..infra.ffmpeg_bootstrap import ensure_ffmpeg, is_portable_ffmpeg_installed
    if not is_portable_ffmpeg_installed():
        ensure_ffmpeg()


def _get_video_dims_ffprobe(input_path):
    ffprobe = get_ffprobe()
    try:
        w = int(subprocess.check_output(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width", "-of", "default=nw=1:nk=1",
             str(input_path)],
            text=True, stderr=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW,
        ).strip())
        h = int(subprocess.check_output(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=height", "-of", "default=nw=1:nk=1",
             str(input_path)],
            text=True, stderr=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW,
        ).strip())
        return w, h
    except FileNotFoundError:
        raise RuntimeError("FFprobe not found. Install FFmpeg: https://ffmpeg.org/download.html")


def _encode_thread_count(out_w, out_h):
    cpu_cores = os.cpu_count() or 4
    return max(2, min(cpu_cores, 4 if out_w * out_h >= 3840 * 2160 else 8))


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


def _build_ffmpeg_pipe(out_w, out_h, fps_val, crf, x264_preset, tune, output_path, extra_vf=None):
    enc_threads = _encode_thread_count(out_w, out_h)
    ffmpeg = get_ffmpeg()
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}", "-pix_fmt", "bgr24",
        "-r", str(fps_val), "-i", "-",
        "-c:v", "libx264", "-crf", str(crf), "-preset", x264_preset,
        "-profile:v", "high", "-level:v", "5.1",
    ]
    if tune:
        cmd += ["-tune", tune]
    if extra_vf:
        cmd += ["-vf", extra_vf]
    cmd += [
        "-pix_fmt", "yuv420p",
        "-x264-params", f"threads={enc_threads}:lookahead-threads=1:rc-lookahead=20",
        "-threads", str(enc_threads),
        "-movflags", "+faststart",
        str(output_path),
    ]
    return cmd


def _build_shader_ffmpeg_cmd(input_path, output_path, out_w, out_h, mode, crf, x264_preset, tune, fit_w=0, fit_h=0):
    enc_threads = _encode_thread_count(out_w, out_h)
    ffmpeg = get_ffmpeg()

    if mode == "light":
        vf = f"scale={out_w}:{out_h}:flags=lanczos"
    elif mode == "strong":
        vf = (f"scale={out_w}:{out_h}:flags=lanczos+accurate_rnd+full_chroma_inp+full_chroma_int,"
              f"unsharp=7:7:1.0:7:7:0.0,smartblur=1.0:0.8:0")
    else:
        vf = (f"scale={out_w}:{out_h}:flags=lanczos+accurate_rnd+full_chroma_inp+full_chroma_int,"
              f"unsharp=5:5:0.8:5:5:0.0,smartblur=0.8:0.5:0")

    if fit_w > 0 and fit_h > 0 and (out_w > fit_w or out_h > fit_h):
        vf += f",scale={fit_w}:{fit_h}:force_original_aspect_ratio=decrease:flags=lanczos"

    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(input_path),
        "-vf", vf,
        "-c:v", "libx264", "-crf", str(crf), "-preset", x264_preset,
        "-profile:v", "high", "-level:v", "5.1",
        "-pix_fmt", "yuv420p",
        "-x264-params", f"threads={enc_threads}:lookahead-threads=1:rc-lookahead=20",
        "-threads", str(enc_threads),
        "-movflags", "+faststart",
    ]
    if tune:
        cmd += ["-tune", tune]
    cmd += [str(output_path)]
    return cmd


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

    ffmpeg_cmd = _build_ffmpeg_pipe(out_w, out_h, fps_val, q["crf"], q["x264"], q.get("tune"), output_path, extra_vf)

    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                                    creationflags=CREATE_NO_WINDOW)
    stderr_lines = []
    def _read_stderr():
        try:
            for line in ffmpeg_proc.stderr:
                decoded = line.decode(errors="replace").strip()
                if decoded:
                    stderr_lines.append(decoded)
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

    _mux_audio(str(output_path), str(input_path))


def _upscale_shader(input_path, output_path, entry, scale, preset, fit_w, fit_h, progress_cb):
    q = QUALITY_PRESETS.get(preset, QUALITY_PRESETS["high"])
    mode = entry.get("default_mode", "medium")
    modes = entry.get("modes", ["medium"])

    _ensure_ffmpeg()

    w, h = _get_video_dims_ffprobe(input_path)
    out_w = w * scale
    out_h = h * scale

    ffmpeg_cmd = _build_shader_ffmpeg_cmd(
        input_path, output_path, out_w, out_h,
        mode, q["crf"], q["x264"], q.get("tune", "animation"),
        fit_w, fit_h,
    )

    if progress_cb:
        progress_cb(20, f"Running shader upscale ({mode} mode, {scale}x)...")

    if progress_cb:
        progress_cb(20, f"Running shader upscale ({mode} mode, {scale}x)...")

    try:
        proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                creationflags=CREATE_NO_WINDOW)

        if progress_cb:
            progress_cb(50, "FFmpeg encoding in progress...")

        out, err = proc.communicate(timeout=7200)
        out, err = proc.communicate(timeout=7200)
        if proc.returncode != 0:
            err_text = (err or out).decode(errors="replace").strip()
            if err_text:
                raise RuntimeError(f"FFmpeg failed (exit {proc.returncode}): {err_text[-500:]}")
            raise RuntimeError(f"FFmpeg failed with exit code {proc.returncode}")
    except subprocess.TimeoutExpired:
        proc.kill()
        raise RuntimeError("Upscale timed out after 2 hours")

    _mux_audio(str(output_path), str(input_path))


def _upscale_onnx(input_path, output_path, model_name, entry, scale, preset, fit_w, fit_h, progress_cb):
    q = QUALITY_PRESETS.get(preset, QUALITY_PRESETS["high"])

    import onnxruntime
    providers = []
    if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    artcnn_dir = os.path.join(get_models_dir(), "artcnn")
    onnx_path = os.path.join(artcnn_dir, entry["file"])

    if not os.path.exists(onnx_path):
        from .weight_loader import download_weights
        download_weights(model_name)

    session = onnxruntime.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name

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

    ffmpeg_cmd = _build_ffmpeg_pipe(out_w, out_h, fps_val, q["crf"], q["x264"], q.get("tune"), output_path, extra_vf)

    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                                    creationflags=CREATE_NO_WINDOW)
    stderr_lines = []
    def _read_stderr():
        try:
            for line in ffmpeg_proc.stderr:
                decoded = line.decode(errors="replace").strip()
                if decoded:
                    stderr_lines.append(decoded)
        except Exception:
            pass
    threading.Thread(target=_read_stderr, daemon=True).start()

    try:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            y, u, v = cv2.split(yuv)
            y_f = y.astype(np.float32) / 255.0
            y_tensor = y_f[np.newaxis, np.newaxis, ...]

            outputs = session.run(None, {input_name: y_tensor})
            y_upscaled = outputs[0][0, 0]

            y_out = np.clip(y_upscaled * 255.0, 0, 255).astype(np.uint8)
            u_out = cv2.resize(u, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
            v_out = cv2.resize(v, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

            yuv_out = cv2.merge([y_out, u_out, v_out])
            result_bgr = cv2.cvtColor(yuv_out, cv2.COLOR_YUV2BGR)

            if ffmpeg_proc.poll() is None:
                try:
                    ffmpeg_proc.stdin.write(result_bgr.tobytes())
                except (BrokenPipeError, OSError):
                    break

            frame_idx += 1
            if progress_cb:
                pct = min(100, int((frame_idx / total_frames) * 100))
                progress_cb(pct, f"ONNX upscaling... {frame_idx}/{total_frames}")

    finally:
        cap.release()
        try:
            ffmpeg_proc.stdin.close()
        except Exception:
            pass
        ffmpeg_proc.wait()

    _mux_audio(str(output_path), str(input_path))


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
    monitor: Optional["SystemMonitor"] = None,
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
        _upscale_shader(input_path, output_path, entry, scale, preset, fit_w, fit_h, progress_cb)

    elif method == "onnx":
        _upscale_onnx(input_path, output_path, model_key, entry, scale, preset, fit_w, fit_h, progress_cb)

    else:
        raise ValueError(f"Unknown method '{method}' for model '{model_key}'")

    if monitor:
        monitor.stop()
