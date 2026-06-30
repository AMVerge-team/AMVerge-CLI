from __future__ import annotations

import gc
import subprocess
import sys
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..infra.binaries import get_ffmpeg
from .registry import QUALITY_PRESETS, get_model
from .weight_loader import is_weight_downloaded, load_weights_if_available

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

INTERPOLATION_AVAILABLE = False
try:
    import torch
    import cv2
    INTERPOLATION_AVAILABLE = True
except ImportError:
    pass


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _frame_to_tensor(frame, device):
    tensor = torch.from_numpy(frame).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device, memory_format=torch.channels_last)


def _tensor_to_frame(tensor):
    t = tensor.detach().squeeze(0).permute(1, 2, 0)
    return (t * 255.0).clamp(0, 255).to(dtype=torch.uint8).cpu().numpy()


def _pad_to_mod(frame, mod=32):
    h, w = frame.shape[:2]
    pad_h = (mod - h % mod) % mod
    pad_w = (mod - w % mod) % mod
    if pad_h == 0 and pad_w == 0:
        return frame
    return cv2.copyMakeBorder(frame, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT)


def _unpad(frame, orig_h, orig_w):
    return frame[:orig_h, :orig_w]


def _encode_thread_count(out_w, out_h):
    cpu_cores = os.cpu_count() or 4
    return max(2, min(cpu_cores, 4 if out_w * out_h >= 3840 * 2160 else 8))


def _build_pipe(out_w, out_h, fps_val, crf, x264_preset, tune, output_path):
    enc_threads = _encode_thread_count(out_w, out_h)
    ffmpeg = get_ffmpeg()
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}", "-pix_fmt", "bgr24",
        "-r", str(fps_val), "-i", "-",
        "-c:v", "libx264", "-crf", str(crf), "-preset", x264_preset,
        "-profile:v", "high",
    ]
    if tune:
        cmd += ["-tune", tune]
    cmd += [
        "-pix_fmt", "yuv420p",
        "-x264-params", f"threads={enc_threads}:lookahead-threads=1:rc-lookahead=20",
        "-threads", str(enc_threads),
        "-movflags", "+faststart",
        str(output_path),
    ]
    return cmd


def _mux_audio(video_path, audio_source_path):
    from ..infra.binaries import get_ffprobe as _get_ffprobe
    ffprobe = _get_ffprobe()
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

    def _mux(audio_codec_args):
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(video_path),
            "-i", str(audio_source_path),
            "-c:v", "copy", *audio_codec_args,
            "-map", "0:v:0", "-map", "1:a:0?",
            "-movflags", "+faststart", tmp,
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=300, creationflags=CREATE_NO_WINDOW)
            if r.returncode == 0:
                os.replace(tmp, str(video_path))
                return True
        except Exception:
            pass
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass
        return False

    import os as _os
    if _mux(["-c:a", "copy"]):
        return True
    return _mux(["-c:a", "aac", "-b:a", "192k"])


def interpolate_video(
    input_path: str,
    output_path: str,
    model_key: str = "rife4.25",
    factor: int = 2,
    preset: str = "high",
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> None:
    entry = get_model(model_key)
    if entry is None:
        raise ValueError(f"Unknown model: {model_key}")

    q = QUALITY_PRESETS.get(preset, QUALITY_PRESETS["high"])

    if not is_weight_downloaded(model_key):
        from .weight_loader import download_weights
        download_weights(model_key, progress_cb=progress_cb)

    device = _get_device()
    model = load_weights_if_available(model_key, str(device))

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open: {input_path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_val = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    out_w = w
    out_h = h
    output_fps = fps_val * factor

    ffmpeg_cmd = _build_pipe(out_w, out_h, output_fps, q["crf"], q["x264"], q.get("tune"), output_path)

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
        ret, prev_frame = cap.read()
        if not ret:
            raise RuntimeError("No frames in video")

        frame_idx = 0
        first_frame = True

        p0 = _frame_to_tensor(cv2.cvtColor(prev_frame, cv2.COLOR_BGR2RGB), device)
        p0_padded = _pad_to_tensor_body(p0)

        if ffmpeg_proc.poll() is None:
            ffmpeg_proc.stdin.write(prev_frame.tobytes())

        while True:
            ret, curr_frame = cap.read()
            if not ret:
                break

            t0 = _frame_to_tensor(cv2.cvtColor(curr_frame, cv2.COLOR_BGR2RGB), device)
            t0_padded = _pad_to_tensor_body(t0)

            model.cachePair(p0_padded, t0_padded)
            saved_f0 = model.flownet.f0
            saved_f1 = model.flownet.f1

            for f in range(1, factor):
                alpha = f / factor
                model.flownet.f0 = saved_f0
                model.flownet.f1 = saved_f1

                with torch.autocast(device_type=str(device), enabled=(device.type == "cuda")):
                    mid = model(p0_padded, t0_padded, alpha)

                mid_unpadded = _unpad_tensor_body(mid, h, w)
                mid_frame = _tensor_to_frame(mid_unpadded)

                if ffmpeg_proc.poll() is None:
                    try:
                        ffmpeg_proc.stdin.write(cv2.cvtColor(mid_frame, cv2.COLOR_RGB2BGR).tobytes())
                    except (BrokenPipeError, OSError):
                        break

                del mid, mid_unpadded, mid_frame

            if ffmpeg_proc.poll() is None:
                try:
                    ffmpeg_proc.stdin.write(curr_frame.tobytes())
                except (BrokenPipeError, OSError):
                    break

            p0 = t0
            p0_padded = t0_padded

            frame_idx += 1
            if frame_idx % 10 == 0:
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            if progress_cb:
                pct = min(100, int((frame_idx / max(1, total_frames - 1)) * 100))
                progress_cb(pct, f"Interpolating... {frame_idx}/{total_frames - 1}")

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    finally:
        cap.release()
        try:
            ffmpeg_proc.stdin.close()
        except Exception:
            pass
        ffmpeg_proc.wait()

    _mux_audio(str(output_path), str(input_path))

    if progress_cb:
        progress_cb(100, "Complete")


def _pad_to_tensor_body(tensor):
    h, w = tensor.shape[2], tensor.shape[3]
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    if pad_h == 0 and pad_w == 0:
        return tensor
    return torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")


def _unpad_tensor_body(tensor, orig_h, orig_w):
    return tensor[:, :, :orig_h, :orig_w]
