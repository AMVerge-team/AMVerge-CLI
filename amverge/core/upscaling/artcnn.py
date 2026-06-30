import os
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..infra.config import get_models_dir
from .registry import QUALITY_PRESETS, get_model, get_onnx_models

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

ARTCNN_MODELS = {
    key: {
        "file": entry["file"],
        "scale": entry["scales"][0],
        "download": entry["url"],
        "sha256": entry.get("hash"),
    }
    for key, entry in get_onnx_models().items()
}


def _get_artcnn_dir():
    return os.path.join(get_models_dir(), "artcnn")


def _download_artcnn_model(model_name, progress_cb=None):
    import urllib.request
    import ssl

    ctx = ssl._create_unverified_context()
    model_info = ARTCNN_MODELS.get(model_name)
    if not model_info:
        raise ValueError(f"Unknown ArtCNN model: {model_name}")

    dest_dir = _get_artcnn_dir()
    os.makedirs(dest_dir, exist_ok=True)

    dest_path = os.path.join(dest_dir, model_info["file"])
    if os.path.exists(dest_path):
        return dest_path

    url = model_info["download"]

    if progress_cb:
        progress_cb(0, f"Downloading {model_name}...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "amverge/1.0"})
        with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk_size = 65536
            with open(dest_path, "wb") as f:
                while True:
                    chunk = resp.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_cb and total > 0:
                        pct = min(99, int(downloaded * 100 / total))
                        progress_cb(pct, f"Downloading {model_name}... {pct}%")
    except Exception as e:
        if os.path.exists(dest_path):
            os.unlink(dest_path)
        raise RuntimeError(f"Failed to download ArtCNN model {model_name}: {e}")

    if progress_cb:
        progress_cb(100, f"Downloaded {model_name}")
    return dest_path


def _get_onnx_inference_session(model_path):
    try:
        import onnxruntime

        providers = []
        if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        return onnxruntime.InferenceSession(model_path, providers=providers)
    except ImportError:
        raise RuntimeError(
            "onnxruntime not installed. Run: pip install onnxruntime-gpu"
        )


def upscale_video_artcnn(
    input_path: str | Path,
    output_path: str | Path,
    model_name: str = "C4F32",
    scale: int = 2,
    preset: str = "high",
    fit_w: int = 0,
    fit_h: int = 0,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> None:
    from ..infra.binaries import get_ffmpeg, get_ffprobe

    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    if model_name not in ARTCNN_MODELS:
        raise ValueError(
            f"Unknown ArtCNN model '{model_name}'. Valid: {list(ARTCNN_MODELS.keys())}"
        )

    q = QUALITY_PRESETS.get(preset, QUALITY_PRESETS["high"])

    try:
        import cv2
    except ImportError:
        raise RuntimeError("ArtCNN upscaling requires opencv. Run: pip install amverge[upscale]")

    model_path = _download_artcnn_model(model_name, progress_cb)

    if progress_cb:
        progress_cb(0, f"Loading ArtCNN model {model_name}...")

    session = _get_onnx_inference_session(model_path)
    input_name = session.get_inputs()[0].name

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open input video: {input_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_val = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 1

    out_w = w * scale
    out_h = h * scale

    cpu_cores = os.cpu_count() or 4
    enc_threads = max(2, min(cpu_cores, 4 if out_w * out_h >= 3840 * 2160 else 8))

    ffmpeg_cmd = [
        get_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}", "-pix_fmt", "bgr24",
        "-r", str(fps_val), "-i", "-",
        "-c:v", "libx264", "-crf", str(q["crf"]), "-preset", q["x264"],
        "-profile:v", "high", "-level:v", "5.1",
    ]
    if q.get("tune"):
        ffmpeg_cmd += ["-tune", q["tune"]]
    if fit_w > 0 and fit_h > 0 and (out_w > fit_w or out_h > fit_h):
        ffmpeg_cmd += ["-vf", f"scale={fit_w}:{fit_h}:force_original_aspect_ratio=decrease:flags=lanczos"]
    ffmpeg_cmd += [
        "-pix_fmt", "yuv420p",
        "-x264-params", f"threads={enc_threads}:lookahead-threads=1:rc-lookahead=20",
        "-threads", str(enc_threads),
        "-movflags", "+faststart",
        str(output_path),
    ]

    import subprocess
    import threading

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

            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            y, u, v = cv2.split(yuv)
            y = y.astype(np.float32) / 255.0
            y_tensor = y[np.newaxis, np.newaxis, ...]

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
                denom = max(total_frames, 1)
                pct = min(100, int((frame_idx / denom) * 100))
                progress_cb(pct, f"ArtCNN upscaling... {frame_idx}/{denom}")

    finally:
        cap.release()
        try:
            ffmpeg_proc.stdin.close()
        except Exception:
            pass
        ffmpeg_proc.wait()

    from .upscale import _mux_audio
    _mux_audio(str(output_path), str(input_path))
