import gc
import os
import subprocess
import threading

import numpy as np

from ..infra.config import get_models_dir
from ..infra.model_store import download_with_resume
from .ffmpeg_helpers import CREATE_NO_WINDOW, build_ffmpeg_pipe, get_color_args, mux_audio
from .registry import QUALITY_PRESETS, get_model


def get_artcnn_dir():
    return os.path.join(get_models_dir(), "artcnn")


def _model_files(entry):
    files = [entry["file"]]
    if entry.get("chroma_file"):
        files.append(entry["chroma_file"])
    return files


def get_artcnn_path(model_key):
    entry = get_model(model_key)
    if entry is None or "file" not in entry:
        raise ValueError(f"Unknown ONNX model: {model_key}")
    return os.path.join(get_artcnn_dir(), entry["file"])


def get_artcnn_chroma_path(model_key):
    entry = get_model(model_key)
    if entry is None or not entry.get("chroma_file"):
        return None
    return os.path.join(get_artcnn_dir(), entry["chroma_file"])


def is_artcnn_downloaded(model_key):
    entry = get_model(model_key)
    if entry is None or "file" not in entry:
        return False
    d = get_artcnn_dir()
    return all(os.path.exists(os.path.join(d, f)) for f in _model_files(entry))


def _download_file(url, dest, label, progress_cb, retries):
    if os.path.exists(dest):
        return True
    try:
        return download_with_resume(
            url,
            dest,
            retries=retries,
            progress_cb=progress_cb,
            progress_formatter=lambda pct, _dl, _tot: f"Downloading {label}... {pct}%",
            done_message=f"Downloaded {label}",
        )
    except Exception as e:
        raise RuntimeError(f"Download failed for {label}: {e}") from e


def download_artcnn(model_key, progress_cb=None, retries=3):
    entry = get_model(model_key)
    if entry is None:
        raise ValueError(f"Unknown ONNX model: {model_key}")

    dest_dir = get_artcnn_dir()
    base_url = entry["url"].rsplit("/", 1)[0]

    for fname in _model_files(entry):
        url = entry["url"] if fname == entry["file"] else f"{base_url}/{fname}"
        _download_file(url, os.path.join(dest_dir, fname), fname, progress_cb, retries)
    return True


def upscale_video_artcnn(input_path, output_path, model_key, entry, scale, preset,
                         fit_w, fit_h, progress_cb=None):
    import cv2
    import onnxruntime

    q = QUALITY_PRESETS.get(preset, QUALITY_PRESETS["high"])

    providers = []
    if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    if not is_artcnn_downloaded(model_key):
        download_artcnn(model_key, progress_cb)

    so = onnxruntime.SessionOptions()
    so.enable_cpu_mem_arena = False
    so.enable_mem_pattern = False

    session = onnxruntime.InferenceSession(get_artcnn_path(model_key), sess_options=so, providers=providers)
    input_name = session.get_inputs()[0].name

    chroma_session = None
    chroma_input = None
    chroma_path = get_artcnn_chroma_path(model_key)
    if chroma_path:
        chroma_session = onnxruntime.InferenceSession(chroma_path, sess_options=so, providers=providers)
        chroma_input = chroma_session.get_inputs()[0].name

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

    ffmpeg_cmd = build_ffmpeg_pipe(out_w, out_h, fps_val, q["crf"], q["x264"],
                                   q.get("tune"), output_path, extra_vf,
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

            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            y, u, v = cv2.split(yuv)
            y_f = y.astype(np.float32) / 255.0
            y_tensor = y_f[np.newaxis, np.newaxis, ...]

            y_upscaled = session.run(None, {input_name: y_tensor})[0][0, 0]
            y_norm = np.clip(y_upscaled, 0.0, 1.0)
            y_out = (y_norm * 255.0).astype(np.uint8)

            if chroma_session is not None:
                u_up = cv2.resize(u, (out_w, out_h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
                v_up = cv2.resize(v, (out_w, out_h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
                chroma_in = np.stack([y_norm, u_up, v_up], axis=0)[np.newaxis, ...]
                chroma_out = chroma_session.run(None, {chroma_input: chroma_in})[0][0]
                u_out = np.clip(chroma_out[0] * 255.0, 0, 255).astype(np.uint8)
                v_out = np.clip(chroma_out[1] * 255.0, 0, 255).astype(np.uint8)
            else:
                u_out = cv2.resize(u, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
                v_out = cv2.resize(v, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

            yuv_out = cv2.merge([y_out, u_out, v_out])
            result_bgr = cv2.cvtColor(yuv_out, cv2.COLOR_YUV2BGR)

            if ffmpeg_proc.poll() is None:
                try:
                    ffmpeg_proc.stdin.write(result_bgr.tobytes())
                except (BrokenPipeError, OSError):
                    break

            del y_upscaled, y_norm, y_out, u_out, v_out, yuv_out, result_bgr, y_tensor, y_f, yuv
            frame_idx += 1
            if frame_idx % 10 == 0:
                gc.collect()
            if progress_cb:
                pct = min(100, int((frame_idx / total_frames) * 100))
                progress_cb(pct, f"ArtCNN upscaling... {frame_idx}/{total_frames}")
    finally:
        cap.release()
        try:
            ffmpeg_proc.stdin.close()
        except Exception:
            pass
        ffmpeg_proc.wait()

    mux_audio(str(output_path), str(input_path))
