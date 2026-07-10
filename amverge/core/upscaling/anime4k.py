import glob
import os
import ssl
import subprocess
import urllib.request
import zipfile

from ..infra.binaries import get_ffmpeg
from ..infra.config import get_models_dir
from .ffmpeg_helpers import (
    CREATE_NO_WINDOW,
    encode_thread_count,
    ensure_ffmpeg,
    get_color_args,
    get_video_dims_ffprobe,
    mux_audio,
)
from .registry import QUALITY_PRESETS, get_model

ANIME4K_MODE_PRESETS = {
    "light": {
        "restore": [],
        "upscale": "Anime4K_Upscale_CNN_x2_S.glsl",
    },
    "medium": {
        "restore": ["Anime4K_Restore_CNN_M.glsl"],
        "upscale": "Anime4K_Upscale_CNN_x2_M.glsl",
    },
    "strong": {
        "restore": ["Anime4K_Clamp_Highlights.glsl", "Anime4K_Restore_CNN_VL.glsl"],
        "upscale": "Anime4K_Upscale_CNN_x2_VL.glsl",
    },
}

_libplacebo_cache = None


def get_shader_dir():
    return os.path.join(get_models_dir(), "anime4k")


def list_shaders():
    return [os.path.basename(p) for p in glob.glob(os.path.join(get_shader_dir(), "*.glsl"))
            if not os.path.basename(p).startswith("_chain_")]


def is_anime4k_downloaded():
    return len(list_shaders()) > 0


def libplacebo_available():
    global _libplacebo_cache
    if _libplacebo_cache is not None:
        return _libplacebo_cache
    try:
        ensure_ffmpeg()
        r = subprocess.run(
            [get_ffmpeg(), "-hide_banner", "-filters"],
            capture_output=True, text=True, timeout=15,
            creationflags=CREATE_NO_WINDOW,
        )
        _libplacebo_cache = "libplacebo" in r.stdout
    except Exception:
        _libplacebo_cache = False
    return _libplacebo_cache


def download_anime4k_shaders(progress_cb=None):
    entry = get_model("anime4k") or {}
    url = entry.get("download_url", "")
    if not url:
        raise RuntimeError("No download URL for anime4k in registry")

    dest_dir = get_shader_dir()
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "Anime4K_v4.0.zip")

    ctx = ssl._create_unverified_context()

    if not os.path.exists(zip_path):
        req = urllib.request.Request(url, headers={"User-Agent": "amverge/1.0"})
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
                        progress_cb(pct, f"Downloading shaders... {pct}%")

    if progress_cb:
        progress_cb(100, "Extracting shaders...")

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.namelist():
            if member.endswith(".glsl"):
                data = zf.read(member)
                out = os.path.join(dest_dir, os.path.basename(member))
                with open(out, "wb") as f:
                    f.write(data)

    return list_shaders()


def _chain_shader_files(mode, scale):
    preset = ANIME4K_MODE_PRESETS.get(mode, ANIME4K_MODE_PRESETS["medium"])
    passes = max(1, scale // 2)
    files = list(preset["restore"])
    files += [preset["upscale"]] * passes
    return files


def _write_combined_shader(mode, scale, dest_dir):
    shader_dir = get_shader_dir()
    files = _chain_shader_files(mode, scale)

    missing = [f for f in files if not os.path.exists(os.path.join(shader_dir, f))]
    if missing:
        raise FileNotFoundError(f"Missing Anime4K shaders: {', '.join(missing)}")

    parts = []
    for f in files:
        with open(os.path.join(shader_dir, f), "r", encoding="utf-8", errors="replace") as fh:
            parts.append(fh.read())

    combined = os.path.join(dest_dir, f"_chain_{mode}_{scale}x.glsl")
    with open(combined, "w", encoding="utf-8") as fh:
        fh.write("\n\n".join(parts))
    return combined


def _fit_dims(w, h, scale, fit_w, fit_h):
    out_w = w * scale
    out_h = h * scale
    if fit_w > 0 and fit_h > 0 and (out_w > fit_w or out_h > fit_h):
        ratio = min(fit_w / out_w, fit_h / out_h)
        out_w = max(2, int(out_w * ratio) // 2 * 2)
        out_h = max(2, int(out_h * ratio) // 2 * 2)
    return out_w, out_h


def _build_libplacebo_cmd(input_path, output_path, out_w, out_h, shader_name,
                          crf, x264_preset, tune, color_args=None):
    enc_threads = encode_thread_count(out_w, out_h)
    cmd = [
        get_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(input_path),
        "-vf", f"libplacebo=w={out_w}:h={out_h}:custom_shader_path={shader_name}",
        "-c:v", "libx264", "-crf", str(crf), "-preset", x264_preset,
        "-profile:v", "high",
        "-pix_fmt", "yuv420p",
        "-x264-params", f"threads={enc_threads}:lookahead-threads=1:rc-lookahead=20",
        "-threads", str(enc_threads),
        "-movflags", "+faststart",
    ]
    if tune:
        cmd += ["-tune", tune]
    if color_args:
        cmd += color_args
    cmd += [str(output_path)]
    return cmd


def _build_fallback_cmd(input_path, output_path, out_w, out_h, mode,
                        crf, x264_preset, tune, color_args=None):
    enc_threads = encode_thread_count(out_w, out_h)

    if mode == "light":
        vf = f"scale={out_w}:{out_h}:flags=lanczos"
    elif mode == "strong":
        vf = (f"scale={out_w}:{out_h}:flags=lanczos+accurate_rnd+full_chroma_inp+full_chroma_int,"
              f"unsharp=7:7:1.0:7:7:0.0,smartblur=1.0:0.8:0")
    else:
        vf = (f"scale={out_w}:{out_h}:flags=lanczos+accurate_rnd+full_chroma_inp+full_chroma_int,"
              f"unsharp=5:5:0.8:5:5:0.0,smartblur=0.8:0.5:0")

    cmd = [
        get_ffmpeg(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(input_path),
        "-vf", vf,
        "-c:v", "libx264", "-crf", str(crf), "-preset", x264_preset,
        "-profile:v", "high",
        "-pix_fmt", "yuv420p",
        "-x264-params", f"threads={enc_threads}:lookahead-threads=1:rc-lookahead=20",
        "-threads", str(enc_threads),
        "-movflags", "+faststart",
    ]
    if tune:
        cmd += ["-tune", tune]
    if color_args:
        cmd += color_args
    cmd += [str(output_path)]
    return cmd


def _run(cmd, timeout, cwd=None):
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            cwd=cwd, creationflags=CREATE_NO_WINDOW)
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise RuntimeError("Anime4K upscale timed out after 2 hours")
    if proc.returncode != 0:
        err_text = (err or out).decode(errors="replace").strip()
        return False, err_text[-500:]
    return True, ""


def upscale_video_anime4k(input_path, output_path, entry, scale, preset,
                          fit_w, fit_h, progress_cb=None):
    q = QUALITY_PRESETS.get(preset, QUALITY_PRESETS["high"])
    mode = entry.get("default_mode", "medium")

    ensure_ffmpeg()

    w, h = get_video_dims_ffprobe(input_path)
    out_w, out_h = _fit_dims(w, h, scale, fit_w, fit_h)
    tune = q.get("tune", "animation")
    color_args = get_color_args(input_path)

    used_shaders = False
    if libplacebo_available():
        if not is_anime4k_downloaded():
            if progress_cb:
                progress_cb(5, "Downloading Anime4K shaders...")
            download_anime4k_shaders(progress_cb)
        dest_dir = os.path.dirname(str(output_path)) or "."
        combined = None
        try:
            combined = _write_combined_shader(mode, scale, dest_dir)
            cmd = _build_libplacebo_cmd(input_path, output_path, out_w, out_h,
                                        os.path.basename(combined), q["crf"], q["x264"], tune,
                                        color_args=color_args)
            if progress_cb:
                progress_cb(30, f"Anime4K shaders ({mode}, {scale}x) via libplacebo...")
            ok, err_text = _run(cmd, timeout=7200, cwd=dest_dir)
            used_shaders = ok
            if not ok and progress_cb:
                progress_cb(35, "libplacebo failed, falling back to lanczos...")
        except FileNotFoundError:
            used_shaders = False
        finally:
            if combined and os.path.exists(combined):
                try:
                    os.remove(combined)
                except OSError:
                    pass

    if not used_shaders:
        cmd = _build_fallback_cmd(input_path, output_path, out_w, out_h,
                                  mode, q["crf"], q["x264"], tune, color_args=color_args)
        if progress_cb:
            progress_cb(40, f"Anime4K fallback ({mode}, {scale}x, lanczos)...")
        ok, err_text = _run(cmd, timeout=7200)
        if not ok:
            raise RuntimeError(f"FFmpeg failed: {err_text}")

    if progress_cb:
        progress_cb(95, "Muxing audio...")
    mux_audio(str(output_path), str(input_path))
    if progress_cb:
        progress_cb(100, "Done")
