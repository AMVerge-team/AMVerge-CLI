from __future__ import annotations

import gc
import os
import ssl
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from typing import Callable, Optional

from ..infra.config import get_amverge_config_dir
from ..infra.binaries import get_ffmpeg, get_ffprobe

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

DEPTH_AVAILABLE = False
try:
    import numpy as np
    import torch
    import cv2
    from depth_anything_v2.dpt import DepthAnythingV2
    DEPTH_AVAILABLE = True
except ImportError:
    pass

MODEL_CONFIGS: dict[str, dict] = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
        "url": "https://github.com/moongetsu/AniSmooth-Models/releases/download/depth/depth_anything_v2_vits.pth",
        "file": "depth_anything_v2_vits.pth",
    },
    "vitb": {
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
        "url": "https://github.com/moongetsu/AniSmooth-Models/releases/download/depth/depth_anything_v2_vitb.pth",
        "file": "depth_anything_v2_vitb.pth",
    },
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
        "url": "https://github.com/moongetsu/AniSmooth-Models/releases/download/depth/depth_anything_v2_vitl.pth",
        "file": "depth_anything_v2_vitl.pth",
    },
}

COLMAPS: dict[str, int] = {}
if DEPTH_AVAILABLE:
    for _name in [
        "inferno", "viridis", "plasma", "magma", "turbo", "jet",
        "twilight", "hot", "cool", "rainbow", "ocean", "bone", "winter", "summer",
    ]:
        _attr = f"COLORMAP_{_name.upper()}"
        if hasattr(cv2, _attr):
            COLMAPS[_name] = getattr(cv2, _attr)


def _get_depth_models_dir() -> str:
    return os.path.join(get_amverge_config_dir(), "models", "depth")


def _get_model_path(encoder: str) -> str:
    config = MODEL_CONFIGS.get(encoder)
    if not config:
        raise ValueError(f"Unknown encoder: {encoder}")
    return os.path.join(_get_depth_models_dir(), config["file"])


def is_model_downloaded(encoder: str) -> bool:
    path = _get_model_path(encoder)
    return os.path.exists(path) and os.path.getsize(path) > 0


def download_model(
    encoder: str,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> None:
    config = MODEL_CONFIGS.get(encoder)
    if not config:
        raise ValueError(f"Unknown encoder: {encoder}. Choose from: {list(MODEL_CONFIGS.keys())}")

    if is_model_downloaded(encoder):
        return

    dest = _get_model_path(encoder)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    temp = dest + ".part"
    url = config["url"]
    ctx = ssl._create_unverified_context()

    for attempt in range(3):
        try:
            existing = os.path.getsize(temp) if os.path.exists(temp) else 0
            headers: dict[str, str] = {"User-Agent": "amverge/1.0"}
            if existing > 0 and attempt > 0:
                headers["Range"] = f"bytes={existing}-"
            elif existing > 0:
                os.remove(temp)
                existing = 0

            req = urllib.request.Request(url, headers=headers)
            resp = urllib.request.urlopen(req, timeout=120, context=ctx)
            code = resp.getcode()
            if code not in (200, 206):
                raise urllib.error.HTTPError(url, code, "", None, None)

            total = int(resp.headers.get("Content-Length", 0))
            total = total + existing if total > 0 and code == 206 else total

            file_mode = "ab" if code == 206 else "wb"
            with open(temp, file_mode) as f:
                downloaded = existing
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if progress_cb and total > 0:
                        pct = min(99, int(downloaded / total * 99))
                        size_mb = downloaded / 1024 / 1024
                        total_mb = total / 1024 / 1024
                        progress_cb(pct, f"Downloading Depth-Anything-V2-{encoder}... {size_mb:.0f}/{total_mb:.0f} MB")

            if os.path.exists(temp) and os.path.getsize(temp) > 0:
                os.replace(temp, dest)
            return

        except Exception:
            if attempt >= 2:
                if os.path.exists(temp):
                    try:
                        os.unlink(temp)
                    except OSError:
                        pass
                raise

    raise RuntimeError(f"Failed to download model {encoder}")


def _mux_audio(video_path: str, audio_source_path: str) -> bool:
    import subprocess

    has_audio = False
    try:
        r = subprocess.run(
            [get_ffprobe(), "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "csv=p=0", audio_source_path],
            capture_output=True, text=True, timeout=10, creationflags=CREATE_NO_WINDOW,
        )
        if "audio" in r.stdout.lower():
            has_audio = True
    except Exception:
        pass

    if not has_audio:
        return False

    tmp = video_path + ".tmp.mp4"
    ffmpeg = get_ffmpeg()

    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-i", audio_source_path,
        "-c:v", "copy", "-c:a", "copy",
        "-map", "0:v:0", "-map", "1:a:0?",
        "-movflags", "+faststart", tmp,
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                           creationflags=CREATE_NO_WINDOW)
        if r.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
            os.replace(tmp, video_path)
            return True
    except Exception:
        pass

    cmd_fallback = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-i", audio_source_path,
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-map", "0:v:0", "-map", "1:a:0?",
        "-movflags", "+faststart", tmp,
    ]
    try:
        r = subprocess.run(cmd_fallback, capture_output=True, text=True, timeout=300,
                           creationflags=CREATE_NO_WINDOW)
        if r.returncode == 0 and os.path.exists(tmp) and os.path.getsize(tmp) > 0:
            os.replace(tmp, video_path)
            return True
    except Exception:
        pass

    if os.path.exists(tmp):
        try:
            os.unlink(tmp)
        except OSError:
            pass
    return False


def generate_depth_map(
    input_path: str,
    output_path: str,
    encoder: str = "vits",
    input_size: int = 518,
    pred_only: bool = False,
    grayscale: bool = False,
    colormap: str = "inferno",
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> None:
    if not DEPTH_AVAILABLE:
        raise ImportError(
            "Depth-Anything V2 not installed.\n"
            "  pip install amverge[depth]"
        )

    if encoder not in MODEL_CONFIGS:
        raise ValueError(f"Unknown encoder: {encoder}. Choose from: {list(MODEL_CONFIGS.keys())}")

    if not is_model_downloaded(encoder):
        if progress_cb:
            progress_cb(0, f"Downloading Depth-Anything-V2-{encoder} model...")
        download_model(encoder, progress_cb=progress_cb)

    config = MODEL_CONFIGS[encoder]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_kwargs = {
        k: v for k, v in config.items()
        if k in ("encoder", "features", "out_channels")
    }
    model = DepthAnythingV2(**model_kwargs)
    model.load_state_dict(torch.load(_get_model_path(encoder), map_location="cpu", weights_only=True))
    model = model.to(device).eval()

    if progress_cb:
        progress_cb(1, f"Model loaded on {device}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open: {input_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    if frame_rate <= 0:
        frame_rate = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    margin_width = 50

    if pred_only:
        output_width = frame_width
    else:
        output_width = frame_width * 2 + margin_width

    cmap_name = colormap if colormap in COLMAPS else "inferno"
    cmap_id = COLMAPS[cmap_name]

    crf = 18
    x264_preset = "medium"
    enc_threads = max(2, min(os.cpu_count() or 4, 8))

    ffmpeg_cmd = [
        get_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{output_width}x{frame_height}", "-pix_fmt", "bgr24",
        "-r", str(frame_rate), "-i", "-",
        "-c:v", "libx264", "-crf", str(crf), "-preset", x264_preset,
        "-profile:v", "high",
        "-pix_fmt", "yuv420p",
        "-x264-params", f"threads={enc_threads}:lookahead-threads=1",
        "-threads", str(enc_threads),
        "-movflags", "+faststart",
        "-an",
        str(output_path),
    ]

    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                                   creationflags=CREATE_NO_WINDOW)

    def _drain_stderr():
        try:
            for _ in ffmpeg_proc.stderr:
                pass
        except Exception:
            pass
    threading.Thread(target=_drain_stderr, daemon=True).start()

    frame_idx = 0
    try:
        while True:
            ret, raw_frame = cap.read()
            if not ret:
                break

            depth = model.infer_image(raw_frame, input_size)

            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)

            if grayscale:
                depth_color = cv2.cvtColor(depth, cv2.COLOR_GRAY2BGR)
            else:
                depth_color = cv2.applyColorMap(depth, cmap_id)

            if pred_only:
                out_frame = depth_color
            else:
                split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
                out_frame = cv2.hconcat([raw_frame, split_region, depth_color])

            if ffmpeg_proc.poll() is not None:
                break
            try:
                ffmpeg_proc.stdin.write(out_frame.tobytes())
            except (BrokenPipeError, OSError):
                break

            frame_idx += 1
            if frame_idx % 50 == 0:
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()

            if progress_cb:
                pct = min(99, max(1, int(frame_idx / max(1, total_frames) * 99)))
                progress_cb(pct, f"Processing depth map... {frame_idx}/{total_frames}")

    finally:
        cap.release()
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        try:
            ffmpeg_proc.stdin.close()
        except Exception:
            pass
        ffmpeg_proc.wait()

    if frame_idx == 0:
        if os.path.exists(output_path):
            os.unlink(output_path)
        raise RuntimeError("No frames processed")

    _mux_audio(str(output_path), str(input_path))

    if progress_cb:
        progress_cb(100, f"Saved: {output_path}")
