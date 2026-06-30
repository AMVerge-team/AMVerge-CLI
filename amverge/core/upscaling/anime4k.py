import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

from ..infra.binaries import get_ffmpeg
from ..infra.config import get_models_dir
from .registry import QUALITY_PRESETS, get_model

CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

ANIME4K_SHADER_FILES = [
    "Anime4K_Clamp_Highlights.glsl",
    "Anime4K_Restore_CNN_VL.glsl",
    "Anime4K_Upscale_CNN_x2_VL.glsl",
    "Anime4K_AutoDownscalePre_x2.glsl",
    "Anime4K_AutoDownscalePre_x4.glsl",
    "Anime4K_Upscale_CNN_x2_M.glsl",
    "Anime4K_Upscale_Deblur_Original_x2.glsl",
    "Anime4K_Upscale_Original_x2.glsl",
    "Anime4K_DarkLines_HQ.glsl",
    "Anime4K_ThinLines_HQ.glsl",
    "Anime4K_Upscale_DTD_x2.glsl",
    "Anime4K_Denoise_Bilateral_Mode.glsl",
    "Anime4K_Deblur_DoG.glsl",
    "Anime4K_DarkLines_VeryFast.glsl",
    "Anime4K_ThinLines_VeryFast.glsl",
    "Anime4K_Upscale_GAN_x2_M.glsl",
    "Anime4K_Upscale_GAN_x2_S.glsl",
    "Anime4K_Upscale_GAN_x2_UL.glsl",
    "Anime4K_Upscale_GAN_x2_VL.glsl",
    "Anime4K_Upscale_GAN_x3_UL.glsl",
    "Anime4K_Upscale_GAN_x3_VL.glsl",
    "Anime4K_Upscale_GAN_x4_UL.glsl",
    "Anime4K_Upscale_GAN_x4_VL.glsl",
    "Anime4K_Restore_CNN_Soft_VL.glsl",
    "Anime4K_Restore_CNN_M.glsl",
    "Anime4K_Restore_CNN_Soft_M.glsl",
]

ANIME4K_DOWNLOAD_URL = (
    "https://github.com/bloc97/Anime4K/releases/download/v4.0.1/Anime4K_v4.0.zip"
)


def _get_anime4k_dir():
    return os.path.join(get_models_dir(), "anime4k")


def _check_libplacebo():
    ffmpeg = get_ffmpeg()
    try:
        r = subprocess.run(
            [ffmpeg, "-hide_banner", "-filters"],
            capture_output=True, text=True, timeout=10,
            creationflags=CREATE_NO_WINDOW,
        )
        if r.returncode == 0 and "libplacebo" in r.stdout:
            return True
    except Exception:
        pass
    return False


ANIME4K_MODE_PRESETS = {
    "light": [
        "Anime4K_Upscale_CNN_x2_VL.glsl",
    ],
    "medium": [
        "Anime4K_DarkLines_VeryFast.glsl",
        "Anime4K_ThinLines_VeryFast.glsl",
        "Anime4K_Upscale_CNN_x2_M.glsl",
    ],
    "strong": [
        "Anime4K_Clamp_Highlights.glsl",
        "Anime4K_Restore_CNN_VL.glsl",
        "Anime4K_Upscale_CNN_x2_VL.glsl",
    ],
}


def _download_anime4k_shaders(progress_cb=None):
    import urllib.request
    import ssl
    import zipfile

    ctx = ssl._create_unverified_context()

    dest_dir = _get_anime4k_dir()
    os.makedirs(dest_dir, exist_ok=True)

    zip_path = os.path.join(dest_dir, "Anime4K_v4.0.zip")
    if not os.path.exists(zip_path):
        if progress_cb:
            progress_cb(0, "Downloading Anime4K shaders...")
        try:
            req = urllib.request.Request(ANIME4K_DOWNLOAD_URL, headers={"User-Agent": "amverge/1.0"})
            with urllib.request.urlopen(req, timeout=120, context=ctx) as resp:
                total = int(resp.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 65536
                with open(zip_path, "wb") as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_cb and total > 0:
                            pct = min(99, int(downloaded * 100 / total))
                            progress_cb(pct, f"Downloading Anime4K shaders... {pct}%")
        except Exception as e:
            raise RuntimeError(f"Failed to download Anime4K shaders: {e}")

        if progress_cb:
            progress_cb(100, "Extracting Anime4K shaders...")

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(dest_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to extract Anime4K shaders: {e}")

    import glob
    present = glob.glob(os.path.join(dest_dir, "*.glsl"))
    return [os.path.basename(p) for p in present]


def _build_anime4k_ffmpeg_cmd(
    input_path, output_path, input_w, input_h,
    scale, mode, crf, x264_preset, tune, max_w=0, max_h=0,
):
    out_w = input_w * scale
    out_h = input_h * scale
    ffmpeg = get_ffmpeg()

    cpu_cores = os.cpu_count() or 4
    enc_threads = max(2, min(cpu_cores, 4 if out_w * out_h >= 3840 * 2160 else 8))

    if mode == "light":
        vf = f"scale={out_w}:{out_h}:flags=lanczos"
    elif mode == "strong":
        vf = (f"scale={out_w}:{out_h}:flags=lanczos+accurate_rnd+full_chroma_inp+full_chroma_int,"
              f"unsharp=7:7:1.0:7:7:0.0,"
              f"smartblur=1.0:0.8:0")
    else:
        vf = (f"scale={out_w}:{out_h}:flags=lanczos+accurate_rnd+full_chroma_inp+full_chroma_int,"
              f"unsharp=5:5:0.8:5:5:0.0,"
              f"smartblur=0.8:0.5:0")

    if max_w > 0 and max_h > 0 and (out_w > max_w or out_h > max_h):
        vf += f",scale={max_w}:{max_h}:force_original_aspect_ratio=decrease:flags=lanczos"

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


def upscale_video_anime4k(
    input_path: str | Path,
    output_path: str | Path,
    scale: int = 2,
    mode: str = "medium",
    preset: str = "high",
    fit_w: int = 0,
    fit_h: int = 0,
    progress_cb: Optional[Callable[[int, str], None]] = None,
) -> None:
    from ...core.upscaling.upscale import _resolve_quality

    input_path = Path(input_path).resolve()
    output_path = Path(output_path).resolve()

    if scale not in (2, 4):
        raise ValueError("Anime4K supports scale 2 or 4")
    if mode not in ANIME4K_MODE_PRESETS:
        raise ValueError(f"Unknown mode '{mode}'. Valid: {list(ANIME4K_MODE_PRESETS.keys())}")

    q = QUALITY_PRESETS.get(preset, QUALITY_PRESETS["high"])

    from ..infra.ffmpeg_bootstrap import ensure_ffmpeg, is_portable_ffmpeg_installed
    if not is_portable_ffmpeg_installed():
        if progress_cb:
            progress_cb(0, "FFmpeg not found. Installing portable FFmpeg...")
        ensure_ffmpeg(progress_cb=progress_cb)

    if progress_cb:
        progress_cb(0, "Checking Anime4K shaders...")

    available = _download_anime4k_shaders(progress_cb)
    if not available:
        raise RuntimeError(
            "No Anime4K shaders available. Download failed or shaders missing."
        )

    from ..infra.binaries import get_ffprobe
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
    except FileNotFoundError:
        raise RuntimeError(
            "FFprobe not found. Install FFmpeg from https://ffmpeg.org/download.html"
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe failed for {input_path}: {e.stderr}")
    try:
        fps_str = subprocess.check_output(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate", "-of", "default=nw=1:nk=1",
             str(input_path)],
            text=True, stderr=subprocess.DEVNULL, creationflags=CREATE_NO_WINDOW,
        ).strip()
        num, den = fps_str.split("/") if "/" in fps_str else (fps_str, "1")
        fps_val = float(num) / float(den)
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        fps_val = 24.0

    if progress_cb:
        progress_cb(10, "Building FFmpeg command...")

    has_libplacebo = _check_libplacebo()
    shader_count = sum(
        1 for s in ANIME4K_MODE_PRESETS[mode]
        if os.path.exists(os.path.join(_get_anime4k_dir(), s))
    )

    ffmpeg_cmd = _build_anime4k_ffmpeg_cmd(
        input_path, output_path, w, h,
        scale, mode, q["crf"], q["x264"], q.get("tune", "animation"),
        fit_w, fit_h,
    )

    if progress_cb:
        progress_cb(20, f"Running Anime4K upscale ({mode} mode, {scale}x)...")

    try:
        proc = subprocess.Popen(
            ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=CREATE_NO_WINDOW,
        )
        out, err = proc.communicate(timeout=7200)

        if proc.returncode != 0:
            err_text = (err or out).decode(errors="replace").strip()
            if err_text:
                raise RuntimeError(f"FFmpeg failed (exit {proc.returncode}): {err_text[-500:]}")
            raise RuntimeError(f"FFmpeg failed with exit code {proc.returncode}")
    except subprocess.TimeoutExpired:
        proc.kill()
        raise RuntimeError("Anime4K upscale timed out after 2 hours")

    if progress_cb:
        progress_cb(100, "Anime4K upscale complete")

    from .upscale import _mux_audio
    _mux_audio(str(output_path), str(input_path))
